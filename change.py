const { EbmlStreamDecoder, EbmlTagId, EbmlTagPosition } = require('ebml-stream');
const { KinesisVideoClient, GetDataEndpointCommand } = require('@aws-sdk/client-kinesis-video');
const { KinesisVideoMedia } = require('@aws-sdk/client-kinesis-video-media');

const WebSocket = require('ws');
const fs = require("fs");
const { S3Client, PutObjectCommand, CopyObjectCommand } = require('@aws-sdk/client-s3');
const { Readable } = require('stream');
const { v4 } = require("uuid");

const os = require('os');
const path = require('path');

const {
  formatPath
} = require('./lca');
const { appsyncCall } = require('./appSyncAPI')

const RECORDING_FILE_PREFIX = formatPath('lca-audio-recordings/');
const RAW_FILE_PREFIX = formatPath('lca-audio-raw/');


let WEBSOCKET_URL;
let CONFIG;

const BlockStream = require('block-stream2');

const BUFFER_SIZE = parseInt(process.env.BUFFER_SIZE || '128', 10);
const stream = require('stream');
const { PassThrough } = require('stream');
const interleave = require('interleave-stream');


const TEMP_FILE_PATH = '/tmp/';

const REGION = 'us-east-1';
let timeToStop = false;
const kvsProducerTimestamp = {};
const kvsServerTimestamp = {};

function sttConfig(flow) {
  switch (flow) {
    case 'nemotron':
      WEBSOCKET_URL = 'wss://whisperstream.exlservice.com:3000/ws/asr';

      CONFIG = {
        type: "start",
        sample_rate: 8000
      };
      break;
    case 'riva':
      WEBSOCKET_URL = "wss://cx-asr.exlservice.com/asr/realtime";
      CONFIG = {
        'service': 'asr',
        'asrPipeline': 'riva',
        'nlpEngine': 'healthcare-agent',
        'ttsEngine': 'polly',
        'tts_emotion_detection': false,
        'user_speaking': true
      }
    default: // nemotron
      WEBSOCKET_URL = 'wss://cx-asr.exlservice.com/asr/realtime-custom-vad';
      // WEBSOCKET_URL = 'wss://whisperstream.exlservice.com:3000/asr/realtime-custom-vad'; //ws://10.90.126.78:443
      // WEBSOCKET_URL = 'wss://whisperstream.exlservice.com:3000/ws/asr'

      CONFIG = {
        backend: "nemotron",
        sample_rate: 8000, 
        type: "start"
      };

      break;
  }
}

function timestampDeltaCheck(n) {
  // Log delta between producer and server timestamps for our two streams.
  const kvsProducerTimestampDelta = Math.abs(
    kvsProducerTimestamp.Caller - kvsProducerTimestamp.Agent,
  );
  const kvsServerTimestampDelta = Math.abs(kvsServerTimestamp.Caller - kvsServerTimestamp.Agent);
  if (kvsProducerTimestampDelta > n) {
    console.log(`WARNING: Producer timestamp delta of received audio is over ${n} seconds.`);
  }
  console.log(
    `Producer timestamps delta: ${kvsProducerTimestampDelta}, Caller: ${kvsProducerTimestamp.Caller}, Agent ${kvsProducerTimestamp.Agent}.`,
  );
  console.log(
    `Server timestamps delta: ${kvsServerTimestampDelta}, Caller: ${kvsServerTimestamp.Caller}, Agent ${kvsServerTimestamp.Agent}.`,
  );
}


async function getForAudio(audioChunks) {
  console.log("Inside GET FOR AUIDO: ", audioChunks);
  const margin = 4;
  var sumLength = 0;

  audioChunks.forEach(chunk => {
    sumLength += chunk.byteLength - margin;
  })

  var sample = new Uint8Array(sumLength);
  var pos = 0;

  audioChunks.forEach(chunk => {
    let tmp = new Uint8Array(chunk.byteLength - margin);
    for (var e = 0; e < chunk.byteLength - margin; e++) {
      tmp[e] = chunk[e + margin];
    }
    sample.set(tmp, pos);
    pos += chunk.byteLength - margin;

  })

  console.log("Returning for getforaudio: ", sample.buffer);
  return sample.buffer;
}



const handler = async function handler(event, context) {
  console.log("EVENT FROM CONNECT _> ", JSON.stringify(event));

  let s3Client;

  let lastMessageTime;
  let currentTrack;
  let currentTrackName;
  let trackDictionary = {};
  let lastFragment;

  let contactId = event.Details.ContactData.ContactId;
  let attributes = event.Details.ContactData.Attributes;
  let sttFlow = Object.entries(attributes).find(([key, value]) => value === "true")?.[0]; // only 1
  // let sttFlow = Object.entries(attributes).filter(([key, value]) => value === "true").map(([key]) => key); // more than 1

  sttConfig(sttFlow);


  let callConnected = true;

  let audioChunksAgent = [];
  let audioChunksCustomer = [];

  let streamArn = event.Details.ContactData.MediaStreams.Customer.Audio.StreamARN;

  let currentPKForAgent = null;
  let currentPKForCustomer = null;

  const kvClient = new KinesisVideoClient({ region: "us-east-1" });
  console.log("KV CLIENT : ", kvClient);

  const getDataCmd = new GetDataEndpointCommand({ APIName: 'GET_MEDIA', StreamARN: streamArn });
  console.log("GET DATA CMD : ", getDataCmd);

  const response2 = await kvClient.send(getDataCmd);
  console.log("RESPONSE : ", response2);
  let firstDecodeEbml = true;


  // const mediaClient = new KinesisVideoArchivedMediaClient({ region: "us-east-1", endpoint: response2.DataEndpoint });

  // let fragmentSelector = { StreamARN: streamArn, FragmentSelector: { StartSelectorType: 'SERVER_TIMESTAMP', TimestampRange: { StartTimestamp :'', EndTimeStamp: ''  } } };

  const mediaClient = new KinesisVideoMedia({ region: "us-east-1", endpoint: response2.DataEndpoint });
  let fragmentSelector = { StreamARN: streamArn, StartSelector: { StartSelectorType: 'NOW' } };
  if (lastFragment && lastFragment.length > 0) {
    fragmentSelector = {
      StreamARN: streamArn,
      StartSelector: {
        StartSelectorType: 'FRAGMENT_NUMBER',
        AfterFragmentNumber: lastFragment,
      },
    };
  }
  console.log("FRAGMENT SELECTOR : ", fragmentSelector);
  const result = await mediaClient.getMedia(fragmentSelector);
  console.log("RESULT HERE : ", result)
  const streamReader = result.Payload;

  const decoder = new EbmlStreamDecoder({
    bufferTagIds: [EbmlTagId.SimpleTag, EbmlTagId.SimpleBlock],
  });

  decoder.on('error', (error) => {
    console.log('Decoder Error:', JSON.stringify(error));
  });


  // const wssCustomer = new WebSocket('ws://10.90.126.78:443');
  const wssCustomer = new WebSocket(WEBSOCKET_URL);

  wssCustomer.on('error on wssCustomer', console.error);

  wssCustomer.on('open', function open() {
    console.log("customer Connection is open, sending data to ASR engine  ");

    wssCustomer.send(JSON.stringify(CONFIG));

    const wssAgent = new WebSocket(WEBSOCKET_URL);

    wssAgent.on('error on wssAgent', console.error);

    wssAgent.on('open', function openAgent() {
      console.log("agent Connection is open, sending data to ASR engine  ");

      wssAgent.send(JSON.stringify(CONFIG));

      decoder.on('data', (chunk) => {
        lastMessageTime = Date().now;
        if (chunk.id === EbmlTagId.Segment && chunk.position === EbmlTagPosition.End) {
          // this is the end of a segment. Lets forcefully stop if needed.

        }
        if (!timeToStop) {
          if (chunk.id === EbmlTagId.TrackNumber) {
            //console.log('TrackNumber', chunk.data);
            currentTrack = parseInt(chunk.data);
          }
          if (chunk.id === EbmlTagId.Name) {
            //console.log('TrackName', chunk.data);
            currentTrackName = chunk.data;
            if (currentTrack && currentTrackName) {
              trackDictionary[currentTrack] = currentTrackName;
            }
          }
          if (chunk.id === EbmlTagId.SimpleTag) {
            if (chunk.Children[0].data === 'AWS_KINESISVIDEO_FRAGMENT_NUMBER') {
              lastFragment = chunk.Children[1].data;
            }
            // capture latest audio timestamps for stream in global variable
            if (chunk.Children[0].data === 'AWS_KINESISVIDEO_SERVER_TIMESTAMP') {
              kvsServerTimestamp['Server'] = chunk.Children[1].data;
            }
            if (chunk.Children[0].data === 'AWS_KINESISVIDEO_PRODUCER_TIMESTAMP') {
              kvsProducerTimestamp['Producer'] = chunk.Children[1].data;
            }
          }
          if (chunk.id === EbmlTagId.SimpleBlock) {
            if (firstDecodeEbml) {
              firstDecodeEbml = false;
              console.log(`decoded ebml, simpleblock size:${chunk.size}`);
              console.log(
                `stream - producer timestamp: ${kvsProducerTimestamp['Producer']}, server timestamp: ${kvsServerTimestamp['Server']}`,
              );
              timestampDeltaCheck(1);
            }
            try {
              if (trackDictionary[chunk.track] === 'AUDIO_TO_CUSTOMER') {
                //handle audio to customer
                // console.log("AGENT CHUNK: >", chunk);

                audioChunksAgent.push(chunk.payload);


                wssAgent.send(chunk.payload);
              }
              if (trackDictionary[chunk.track] === 'AUDIO_FROM_CUSTOMER') {
                // console.log("CUSTOMER CHUNK: >", chunk);


                audioChunksCustomer.push(chunk.payload);


                wssCustomer.send(chunk.payload);
                //handle audio from customer
              }
            } catch (error) {
              console.error('Error posting payload chunk', error);
            }
          }
        }
      }); // use this to find last fragment tag we received

      decoder.on('end', () => {
        // close stdio
        console.log('Finished');
        console.log(`Last fragment ${lastFragment} total size: ${totalSize}`);
        wssCustomer.close()
        wssAgent.close()
      });
    });

    wssAgent.on('message', async function agentEvent(message) {
      const parsedMsg = JSON.parse(message.toString());
      console.log("Agent Socket Event: ", parsedMsg);
      if (parsedMsg.text && parsedMsg.text.length > 0) {
        if (parsedMsg.type === 'final') {
          await appsyncCall({ PK: currentPKForAgent || v4(), CallId: contactId, Channel: 'AGENT', Transcript: parsedMsg.text, IsFinal: true })
          currentPKForAgent = null;
        } else {
          if (!currentPKForAgent) {
            currentPKForAgent = v4();
          }
          await appsyncCall({ PK: currentPKForAgent, CallId: contactId, Channel: 'AGENT', Transcript: parsedMsg.text, IsFinal: false })
        }
      }
    });

  });

  wssCustomer.on('message', async function customerEvent(message) {
    const parsedMsg = JSON.parse(message.toString());
    console.log("Customer Socket Event: ", parsedMsg);
    if (parsedMsg.text && parsedMsg.text.length > 0) {
      if (parsedMsg.type === 'final') { // final transcripts
        await appsyncCall({ PK: currentPKForCustomer || v4(), CallId: contactId, Channel: 'CUSTOMER', Transcript: parsedMsg.text, IsFinal: true })
        currentPKForCustomer = null;
      } else { // partial transcripts
        if (!currentPKForCustomer) {
          currentPKForCustomer = v4();
        }
        await appsyncCall({ PK: currentPKForCustomer, CallId: contactId, Channel: 'CUSTOMER', Transcript: parsedMsg.text, IsFinal: false })
      }
    }
  });

  let firstKvsChunk = true;
  let totalSize = 0;


  try {
    for await (const chunk of streamReader) {
      if (firstKvsChunk) {
        firstKvsChunk = false;
        console.log(`received chunk size: ${chunk.length}`);
      }
      totalSize += chunk.length;
      decoder.write(chunk);

    }
  } catch (error) {
    console.error('error writing to decoder', error);
  } finally {
    console.log('Closing buffers');
    decoder.end();
  }

  const rawAgent = await getForAudio(audioChunksAgent);
  const wavAgent = await Converter.createWav(rawAgent, 8000);

  const rawCustomer = await getForAudio(audioChunksCustomer);
  const wavCustomer = await Converter.createWav(rawCustomer, 8000);

  s3Client = new S3Client({ region: REGION });

  const uploadParamsAgent = {
    Bucket: "call-rec-connect",
    Key: contactId + '-agent.wav',
    Body: wavAgent.buffer
  };

  const uploadParamsCustomer = {
    Bucket: "call-rec-connect",
    Key: contactId + '-customer.wav',
    Body: wavCustomer.buffer
  };

  try {
    console.log("TO WRITE TO S3");
    let s3Data = await s3Client.send(new PutObjectCommand(uploadParamsAgent));
    if (s3Data) {
      console.log("Write to s3 successful agent wav file: " + contactId + "-agent.wav");
      let s3DataCustomer = await s3Client.send(new PutObjectCommand(uploadParamsCustomer));
      if (s3DataCustomer) {
        console.log("Write to s3 successful customer wav file: " + contactId + "-customer.wav");

      }

    }
  } catch (err) {
    console.error('S3 upload error: ', err);
  }
};

// const handler = async function handler(event, context) {
//   const response = await appsyncCall(event);
//   console.log("response: ", response);
// };

class Converter {
  // Convert the raw audio into wav format
  static createWav(samples, sampleRate) {
    const len = samples.byteLength;
    const view = new DataView(new ArrayBuffer(44 + len));
    this._writeString(view, 0, "RIFF");
    view.setUint32(4, 32 + len, true);
    this._writeString(view, 8, "WAVE");
    this._writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    this._writeString(view, 36, "data");
    view.setUint32(40, len, true);
    let offset = 44;
    const srcView = new DataView(samples);
    for (var i = 0; i < len; i += 4, offset += 4) {
      view.setInt32(offset, srcView.getUint32(i));
    }
    return view;
  }

  static _writeString(view, offset, string) {
    for (var i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
}

exports.handler = handler;

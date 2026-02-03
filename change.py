inputs = self.processor(
    silence,
    sampling_rate=self.sr,
    return_tensors="pt",
)

inputs = {
    k: v.to(
        device=self.model.device,
        dtype=self.model.dtype   # 🔑 FIX
    )
    for k, v in inputs.items()
}

_ = self.model.generate(**inputs)



inputs = self.engine.processor(
    audio,
    sampling_rate=self.engine.sr,
    return_tensors="pt",
)
self.utt_preproc += (time.perf_counter() - t0)

inputs = {
    k: v.to(
        device=self.engine.model.device,
        dtype=self.engine.model.dtype   # 🔑 FIX
    )
    for k, v in inputs.items()
}

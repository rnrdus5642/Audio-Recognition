class PcmRecorder extends AudioWorkletProcessor {
  process(inputs) {
    const mono = inputs[0]?.[0];
    if (mono) { const chunk = new Float32Array(mono); this.port.postMessage(chunk, [chunk.buffer]); }
    return true;
  }
}
registerProcessor('pcm-recorder', PcmRecorder);

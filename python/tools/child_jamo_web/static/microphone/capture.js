class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.active = true;
    this.port.onmessage = ({data}) => {
      if (data === 'stop') { this.active = false; this.port.postMessage({stopped:true}); }
    };
  }
  process(inputs) {
    if (this.active && inputs[0]?.[0]?.length) {
      const samples = new Float32Array(inputs[0][0]);
      this.port.postMessage({samples}, [samples.buffer]);
    }
    return true;
  }
}
registerProcessor('microphone-capture', Capture);

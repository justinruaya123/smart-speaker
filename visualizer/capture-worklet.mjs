/* Microphone PCM stays local. No speech service, recording upload or monitoring. */
class VCMCapture extends AudioWorkletProcessor {
  process(inputs,outputs) {
    for(const output of outputs)for(const channel of output)channel.fill(0);
    const input=inputs[0]?.[0];
    if(input?.length){const copy=input.slice();this.port.postMessage({pcm:copy,frame:currentFrame,rate:sampleRate},[copy.buffer]);}
    return true;
  }
}
registerProcessor('vcm-capture',VCMCapture);

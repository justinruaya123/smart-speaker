// Deliberate gate: a transport mock cannot establish anything about ONNX inference.
if (process.argv.includes('--help')) {
  console.log('Real ONNX sanity is deferred until the service protocol and model fixtures are supplied. See docs/onnx-sanity.md. This command never contacts a service or runs a mock.');
} else {
  console.error('NOT RUN: no real inference adapter/model contract is available. Supply the items in docs/onnx-sanity.md first. Mock output is not ONNX validation.');
  process.exitCode = 2;
}

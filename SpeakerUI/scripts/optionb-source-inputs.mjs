/** Additive portable verifier arguments; historical parity helpers stay frozen. */
/** @param {string} [protocol] */
export function preflightArgs(protocol) {
  const args=['-m','scripts.test.optionb_release_check','--preflight'];
  if(protocol !== undefined) args.push('--protocol',protocol);
  return args;
}

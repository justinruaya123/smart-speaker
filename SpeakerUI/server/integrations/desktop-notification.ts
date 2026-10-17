import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const run=promisify(execFile);

/** Local MacBook notification; arbitrary text is passed as data, never source. */
export async function notifyDesktop(message:string,title:string){
  if(process.platform!=='darwin')return false;
  await run('/usr/bin/osascript',['-e','on run argv\ndisplay notification (item 1 of argv) with title (item 2 of argv)\nend run',message,title],{timeout:5000});
  return true;
}

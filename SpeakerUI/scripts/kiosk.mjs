import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
if(existsSync('.env'))process.loadEnvFile('.env');
const binary=process.env.CHROMIUM_PATH||'chromium';
const url=process.env.DISPLAY_ORIGIN||`http://${process.env.FRONTEND_HOST||'127.0.0.1'}:${process.env.FRONTEND_PORT||'3000'}`;
const browser=spawn(binary,['--kiosk','--no-first-run','--noerrdialogs','--disable-session-crashed-bubble','--window-size=800,480',url],{stdio:'inherit'});
browser.on('error',error=>{console.error(`Chromium could not start: ${error.message}. Set CHROMIUM_PATH to the installed browser.`);process.exitCode=1;});
for(const signal of ['SIGINT','SIGTERM'])process.once(signal,()=>browser.kill(signal));
browser.on('exit',code=>{process.exitCode=code||0;});

"""Bounded streaming NDJSON over local TCP; a tool server owns action execution."""
import asyncio
import hmac
import json
from pathlib import Path
import re
from .bundle import load_bundle,below


class Server:
    def __init__(self,manager,releases,*,token=None,allow_diagnostic=False):
        self.manager=manager;self.releases=Path(releases);self.token=token
        self.allow_diagnostic=allow_diagnostic;self.connections=0;self.reload_lock=asyncio.Lock()

    async def handle(self,reader,writer):
        owned=None;authorized=self.token is None
        self.connections+=1
        async def send(value):
            writer.write((json.dumps(value,allow_nan=False,separators=(',',':'))+'\n').encode())
            await asyncio.wait_for(writer.drain(),timeout=5)
        try:
            if self.connections>8:raise ValueError('Connection limit')
            await send(dict(type='hello',protocol='pcv-stream-v1',authentication_required=not authorized,
                provider='CPUExecutionProvider',threads=1,actions_executed=False))
            while True:
                line=await asyncio.wait_for(reader.readline(),timeout=30)
                if not line:break
                if len(line)>16384:raise ValueError('Frame too large')
                def invalid(_):raise ValueError('Nonfinite JSON')
                message=json.loads(line,parse_constant=invalid)
                if not isinstance(message,dict):raise ValueError('Object required')
                kind=message.get('type')
                if not authorized:
                    supplied=message.get('token')
                    if kind!='auth' or not isinstance(supplied,str) or not hmac.compare_digest(supplied,self.token):
                        raise ValueError('Authentication failed')
                    authorized=True;await send(dict(type='authenticated'));continue
                if kind=='start':
                    if owned is not None:raise ValueError('Stream already started')
                    result=self.manager.start(message.get('stream_id'));owned=result['stream_id']
                elif kind in ('audio','end','cancel'):
                    if owned is None or message.get('stream_id',owned)!=owned:raise ValueError('Connection does not own stream')
                    if kind=='audio':
                        result=self.manager.audio(owned,message)
                        if self.manager.active is None:owned=None
                    elif kind=='end':result=self.manager.end(owned);owned=None
                    else:self.manager.cancel(owned);result=dict(type='cancelled',stream_id=owned);owned=None
                elif kind=='status':
                    result=dict(type='status',binding=self.manager.current['binding'],busy=self.manager.active is not None,
                        diagnostic=self.manager.current['diagnostic'])
                elif kind=='reload':
                    release=message.get('release')
                    if not isinstance(release,str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',release):raise ValueError('Invalid release ID')
                    async with self.reload_lock:
                        if self.manager.active is not None and self.manager.active['snapshot'] is not self.manager.current:
                            raise ValueError('One reload is already pending behind the active stream')
                        try:
                            replacement=await asyncio.to_thread(load_bundle,below(self.releases,release),allow_diagnostic=self.allow_diagnostic)
                        except Exception as exc:
                            await send(dict(type='reload_failed',binding=self.manager.current['binding'],reason=str(exc)))
                            continue
                        self.manager.stage(replacement)
                    result=dict(type='reloaded',binding=replacement['binding'],active_stream_pinned=owned is not None or self.manager.active is not None)
                elif kind=='rollback':
                    async with self.reload_lock:binding=self.manager.rollback()
                    result=dict(type='rolled_back',binding=binding)
                else:raise ValueError('Unsupported message type')
                await send(result)
        except Exception as exc:
            try:await send(dict(type='error',accepted=False,command=None,reason=str(exc)))
            except (ConnectionError,TimeoutError):pass
        finally:
            if owned is not None:self.manager.cancel(owned)
            self.connections-=1;writer.close()
            try:await writer.wait_closed()
            except ConnectionError:pass


async def serve(manager,releases,host='127.0.0.1',port=8765,*,token=None,allow_diagnostic=False):
    if host not in ('127.0.0.1','::1','localhost') and not token:raise ValueError('Non-loopback listening requires PCV_SERVER_TOKEN')
    server=Server(manager,releases,token=token,allow_diagnostic=allow_diagnostic)
    listener=await asyncio.start_server(server.handle,host,port,limit=16384,backlog=8)
    print(json.dumps(dict(status='serving',host=host,port=port,protocol='pcv-stream-v1',cpu_only=True)),flush=True)
    async with listener:await listener.serve_forever()

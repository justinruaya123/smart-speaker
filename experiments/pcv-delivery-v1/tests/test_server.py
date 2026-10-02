import asyncio
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
from scripts.runtime.pcv_deploy.stream import StreamManager
from scripts.runtime.pcv_deploy.server import Server
from test_stream import snapshot,audio


def test_real_tcp_stream_ownership_disconnect_and_failed_reload(tmp_path):
    async def check():
        manager=StreamManager(snapshot('fixture'));server=Server(manager,tmp_path)
        listener=await asyncio.start_server(server.handle,'127.0.0.1',0,limit=16384)
        port=listener.sockets[0].getsockname()[1]
        async def connect():
            reader,writer=await asyncio.open_connection('127.0.0.1',port)
            assert json.loads(await reader.readline())['protocol']=='pcv-stream-v1'
            return reader,writer
        async def ask(reader,writer,message):
            writer.write((json.dumps(message)+'\n').encode());await writer.drain()
            return json.loads(await reader.readline())
        r,w=await connect();r2,w2=await connect()
        assert (await ask(r,w,dict(type='start',stream_id='a')))['type']=='started'
        assert (await ask(r,w,dict(type='reload',release='missing-release')))['type']=='reload_failed'
        assert manager.active['id']=='a' and manager.current['binding']=='fixture'
        assert (await ask(r2,w2,dict(type='audio',stream_id='a',**{k:v for k,v in audio().items() if k!='type'})))['type']=='error'
        assert (await ask(r,w,audio()))['events'][0]['accepted']
        w.close();await w.wait_closed()
        for _ in range(20):
            if manager.active is None:break
            await asyncio.sleep(.01)
        assert manager.active is None
        r3,w3=await connect()
        assert (await ask(r3,w3,dict(type='reload',release='../escape')))['type']=='error'
        assert manager.current['binding']=='fixture'
        w2.close();w3.close();await w2.wait_closed();await w3.wait_closed()
        listener.close();await listener.wait_closed()
    asyncio.run(check())

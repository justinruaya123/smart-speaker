import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createOpenWeather } from './openweather.js';

test('weather uses configured coordinates, Celsius and a server-only key', async()=>{
  const weather=createOpenWeather({enabled:true,apiKey:'private-key',latitude:14.5995,longitude:120.9842},async input=>{
    const url=new URL(String(input));assert.equal(url.origin,'https://api.openweathermap.org');
    assert.equal(url.searchParams.get('appid'),'private-key');assert.equal(url.searchParams.get('units'),'metric');
    assert.equal(url.searchParams.get('lat'),'14.5995');
    return Response.json({name:'Manila',dt:1700000000,main:{temp:28.4,feels_like:31.2,humidity:70},weather:[{description:'broken clouds'}]});
  });
  const result=await weather.current();assert.equal(result.temperature,28.4);
  assert.match(result.text,/Manila.*28 degrees Celsius.*broken clouds/);
  assert.ok(!JSON.stringify(result).includes('private-key'));
});
test('disabled, failed and malformed weather never invent a forecast or expose secrets',async()=>{
  const config={enabled:true,apiKey:'secret',latitude:0,longitude:0};
  await assert.rejects(createOpenWeather({...config,enabled:false},async()=>assert.fail('network')).current(),/not configured/i);
  await assert.rejects(createOpenWeather(config,async()=>{throw Error('secret URL');}).current(),/unavailable/i);
  await assert.rejects(createOpenWeather(config,async()=>Response.json({main:{temp:null}})).current(),/invalid/i);
});

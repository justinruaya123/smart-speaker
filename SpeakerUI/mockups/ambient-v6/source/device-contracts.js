// Pure request descriptions only. The mockup never sends these requests.
export const LIGHT_COLORS = Object.freeze({
  Red:{hex:'#ff625d',rgb_color:[255,0,0]},
  Green:{hex:'#68e899',rgb_color:[0,255,0]},
  Blue:{hex:'#649dff',rgb_color:[0,80,255]},
  Orange:{hex:'#ffb361',rgb_color:[255,140,0]},
  'Warm white':{hex:'#ffd6a1',color_temp_kelvin:2700},
  White:{hex:'#eff7ef',color_temp_kelvin:4000},
  'Cool white':{hex:'#c4e8ff',color_temp_kelvin:6500}
});
export function supportsColor(light,name){const c=LIGHT_COLORS[name];return Boolean(c&&(c.rgb_color?light.supported_color_modes.includes('rgb'):light.supported_color_modes.includes('color_temp')));}
export function colorPayload(light,name){const c=LIGHT_COLORS[name];if(!supportsColor(light,name))throw new Error('Unsupported light color');return c.rgb_color?{rgb_color:[...c.rgb_color]}:{color_temp_kelvin:Math.max(light.min_color_temp_kelvin,Math.min(light.max_color_temp_kelvin,c.color_temp_kelvin))};}
export function serviceRequest(intent,p){
  const entity_id=p.entity_id??p.entityId;
  if(intent==='display.temperature_sensor'&&typeof entity_id==='string')return{method:'GET',path:'/api/states/'+encodeURIComponent(entity_id)};
  const light=(service,body)=>({method:'POST',path:'/api/services/light/'+service,body});
  if(intent==='light.toggle')return light('toggle',{entity_id});
  if(intent==='light.on'||intent==='light.off')return light(intent==='light.on'?'turn_on':'turn_off',{entity_id});
  if(intent==='light.group_off')return light('turn_off',{entity_id:p.entity_ids??p.entityIds});
  if(intent==='light.brightness'){const brightness_pct=p.brightness_pct??p.brightness;return light(brightness_pct===0?'turn_off':'turn_on',brightness_pct===0?{entity_id}:{entity_id,brightness_pct});}
  if(intent==='light.color'){const c=LIGHT_COLORS[p.color];return light('turn_on',{entity_id,...(p.rgb_color?{rgb_color:p.rgb_color}:p.color_temp_kelvin?{color_temp_kelvin:p.color_temp_kelvin}:c?.rgb_color?{rgb_color:c.rgb_color}:c?{color_temp_kelvin:c.color_temp_kelvin}:{})});}
  if(intent==='climate.set_temperature')return{method:'POST',path:'/api/services/climate/set_temperature',body:{entity_id:entity_id??'climate.living_room',temperature:p.temperature??p.target}};
  if(intent==='media.shuffle')return{method:'PUT',path:'/v1/me/player/shuffle',query:{state:p.state}};
  if(intent==='media.repeat')return{method:'PUT',path:'/v1/me/player/repeat',query:{state:p.state}};
  if(intent==='media.seek')return{method:'PUT',path:'/v1/me/player/seek',query:{position_ms:p.position_ms}};
  return null;
}

// Local Home Assistant state fixtures. No credentials or network access are used.
export const TEMPERATURE_SENSORS = [
 {entity_id:'sensor.living_room_temperature',state:'26',attributes:{friendly_name:'Living room',device_class:'temperature',unit_of_measurement:'°C'}},
 {entity_id:'sensor.bedroom_temperature',state:'23',attributes:{friendly_name:'Bedroom',device_class:'temperature',unit_of_measurement:'°C'}},
 {entity_id:'sensor.porch_temperature',state:'82.4',attributes:{friendly_name:'Porch',device_class:'temperature',unit_of_measurement:'°F'}},
 {entity_id:'sensor.utility_temperature',state:'unavailable',attributes:{friendly_name:'Utility room',device_class:'temperature',unit_of_measurement:'°C'}}
];
export function temperatureCelsius(sensor){
 if(!sensor||sensor.attributes?.device_class!=='temperature'||typeof sensor.state!=='string'||!sensor.state.trim())return null;
 const value=Number(sensor.state);if(!Number.isFinite(value))return null;
 const unit=sensor.attributes.unit_of_measurement;
 const c=unit==='°C'?value:unit==='°F'?(value-32)*5/9:unit==='K'?value-273.15:null;
 return c===null?null:Math.round(c*10)/10;
}

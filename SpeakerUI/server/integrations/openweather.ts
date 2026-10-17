import { IntegrationError, request, requireSuccess } from './http.js';

export type WeatherConfig = {enabled:boolean;apiKey:string;latitude:number;longitude:number};
export function createOpenWeather(config:WeatherConfig,fetcher:typeof fetch=fetch){
  return {async current(){
    if(!config.enabled||!config.apiKey)throw new IntegrationError('disabled','OpenWeather is not configured. Set OPENWEATHER_ENABLED and OPENWEATHER_API_KEY.');
    const url=new URL('https://api.openweathermap.org/data/2.5/weather');
    url.search=new URLSearchParams({lat:String(config.latitude),lon:String(config.longitude),appid:config.apiKey,units:'metric',lang:'en'}).toString();
    const response=await request(fetcher,url.toString());await requireSuccess(response);
    let data:any;try{data=await response.json();}catch{throw new IntegrationError('invalid_response','OpenWeather returned invalid weather data.');}
    if(!Number.isFinite(data?.main?.temp)||typeof data?.name!=='string'||!data.name.trim()||typeof data?.weather?.[0]?.description!=='string')
      throw new IntegrationError('invalid_response','OpenWeather returned invalid weather data.');
    const location=data.name.slice(0,100),description=data.weather[0].description.slice(0,150),temperature=data.main.temp as number;
    return {provider:'OpenWeather',location,temperature,description,observedAt:Number.isFinite(data.dt)?data.dt*1000:null,
      text:`In ${location}, it is ${Math.round(temperature)} degrees Celsius with ${description}.`};
  }};
}

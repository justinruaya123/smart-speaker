import { isRecord } from '../../shared/protocol.js';
import { IntegrationError, request, requireSuccess } from './http.js';
import { provenance } from '../inference/synthetic-outcomes.js';

export type OpenWeatherConfig = { enabled: false } | { enabled: true; apiKey: string; latitude: number; longitude: number; location: string };
const ttl = 600_000;
export function syntheticWeather() {
  return { status: 'synthetic', detail: 'Synthetic — no device changed. Illustrative weather, not a current observation.',
    provenance: provenance('synthetic'), synthetic: true, provider_accepted:false, device_state_confirmed:false,
    weather: { temperature:28,unit:'°C',state:'illustrative clear',location:'Illustrative sample',observed_at:null as string | null,fetched_at:null as string | null,age_seconds:null as number | null,stale:false } };
}
type WeatherResult = ReturnType<typeof syntheticWeather>;

/** Current-weather v2.5, metric units. dt is the provider's UTC calculation time.
 * Contract: https://openweathermap.org/api/current . Keys/coordinates never leave this module. */
export function createOpenWeather(config: OpenWeatherConfig, fetcher: typeof fetch, now = Date.now) {
  let cache: WeatherResult | undefined, inFlight: Promise<WeatherResult> | undefined;
  function cached(): WeatherResult {
    const value = structuredClone(cache!);
    const age = Math.max(0, Math.floor((now() - Date.parse(value.weather.fetched_at!)) / 1000));
    return { ...value,status:'cached',detail:'Cached OpenWeather snapshot; original observation and retrieval times retained.',
      provenance:{ ...value.provenance,source:'cached' },weather:{ ...value.weather,age_seconds:age,stale:age * 1000 >= ttl } };
  }
  async function retrieve(): Promise<WeatherResult> {
    if (!config.enabled) return syntheticWeather();
    try {
      const url = new URL('https://api.openweathermap.org/data/2.5/weather');
      url.search = new URLSearchParams({ lat:String(config.latitude),lon:String(config.longitude),appid:config.apiKey,units:'metric' }).toString();
      const response = await request(fetcher,url.href); await requireSuccess(response);
      const value: unknown = await response.json();
      if (!isRecord(value) || !isRecord(value.main) || typeof value.main.temp !== 'number' || !Number.isFinite(value.main.temp) ||
        value.main.temp < -100 || value.main.temp > 70 || typeof value.dt !== 'number' || !Number.isSafeInteger(value.dt) || value.dt <= 0 ||
        value.dt * 1000 > now() + 300_000 || !Array.isArray(value.weather) || !isRecord(value.weather[0]) || typeof value.weather[0].description !== 'string' || !value.weather[0].description.trim())
        throw new IntegrationError('invalid_response','OpenWeather returned an invalid observation.');
      const observed_at = new Date(value.dt * 1000).toISOString(), fetched_at = new Date(now()).toISOString();
      cache = { status:'observed',detail:'OpenWeather observation; retrieval and observation times shown separately.',synthetic:false,provider_accepted:false,device_state_confirmed:false,
        provenance:{...provenance('openweather'),observed_at,fetched_at},
        weather:{temperature:value.main.temp,unit:'°C',state:value.weather[0].description.slice(0,100),location:config.location,observed_at,fetched_at,age_seconds:0,stale:false} };
      return structuredClone(cache);
    } catch (error) {
      if (error instanceof IntegrationError && error.outcomeUnknown) throw error;
      return cache ? cached() : syntheticWeather();
    }
  }
  return { read(): Promise<WeatherResult> {
    if (cache && now() - Date.parse(cache.weather.fetched_at!) < ttl) return Promise.resolve(cached());
    inFlight ??= retrieve().finally(() => { inFlight = undefined; });
    return inFlight.then(value => structuredClone(value));
  } };
}

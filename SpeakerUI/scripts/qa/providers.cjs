const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const base = process.env.TEST_BASE_URL || 'http://127.0.0.1:4173';
const out = path.resolve(process.env.TEST_OUTPUT_DIR || 'docs/testing/providers');
fs.mkdirSync(out, { recursive: true });
(async () => {
  const browser = await chromium.launch({ headless: true, ...(process.env.CHROMIUM_EXECUTABLE ? { executablePath: process.env.CHROMIUM_EXECUTABLE } : {}), args: ['--enable-unsafe-swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 800, height: 480 }, hasTouch: true });
  const errors = [], checks = [], commands = [];
  page.on('pageerror', error => errors.push(error.message));
  page.setDefaultTimeout(7000);
  const firstArtwork = 'https://i.scdn.co/image/qa-cover-first', nextArtwork = 'https://i.scdn.co/image/qa-cover-next';
  await page.route('https://i.scdn.co/image/qa-cover-*', route => {
    if (route.request().url().endsWith('broken')) return route.fulfill({ status: 404, body: '' });
    const color = route.request().url() === firstArtwork ? '#316777' : '#986348';
    return route.fulfill({ contentType: 'image/svg+xml', body: `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300"><rect width="300" height="300" fill="${color}"/><circle cx="150" cy="128" r="83" fill="#ceddd7"/><path d="M0 240 95 143 184 229 241 173 300 240V300H0Z" fill="#142a35"/></svg>` });
  });
  const ha = { enabled: true, actionsEnabled: true, status: 'connected', entities: [
    { id: 'light.hue_color_lamp_1', name: 'Ceiling light', state: 'off', stale: false, updatedAt: new Date().toISOString(), attributes: { brightness: 128, color_mode: 'hs', rgb_color: [255, 0, 0], supported_color_modes: ['hs', 'color_temp'], min_color_temp_kelvin: 2000, max_color_temp_kelvin: 6500 } },
    { id: 'sensor.room', name: 'Room temperature', state: '77', stale: false, updatedAt: new Date().toISOString(), attributes: { device_class: 'temperature', unit_of_measurement: '°F' } },
  ] };
  const spotify = { enabled: true, actionsEnabled: true, status: 'connected', updatedAt: Date.now(), playback: {
    playing: true, positionMs: 20000, shuffle: false, repeat: 'off',
    track: { title: 'Live track', artist: 'Live artist', album: 'Live album', artworkUrl: firstArtwork, durationMs: 240000 }, device: { name: 'Desk speaker', volume: 30 },
  } };
  let failLight = false, failTransport = false, delayedSpotify = false;
  await page.route('**/api/home-assistant/**', async route => {
    if (route.request().method() === 'POST') {
      const command = route.request().postDataJSON(); commands.push({ provider: 'ha', ...command });
      if (failLight) return route.fulfill({ status: 502, json: { message: 'Device unavailable' } });
      const light = ha.entities.find(e => e.id === command.entityId);
      if (command.action === 'off') light.state = 'off';
      else { light.state = 'on'; if (command.action === 'brightness') { light.attributes.brightness = Math.round(command.value * 255 / 100); if (!command.value) light.state = 'off'; }
        if (command.action === 'color') { light.attributes.rgb_color = command.rgb; light.attributes.color_mode = 'hs'; }
        if (command.action === 'temperature') { light.attributes.color_temp_kelvin = command.value; light.attributes.color_mode = 'color_temp'; }
      }
    }
    if (failTransport) return route.fulfill({ status: 503, json: {} });
    await route.fulfill({ json: ha });
  });
  await page.route('**/api/spotify/**', async route => {
    if (route.request().method() === 'POST') {
      const command = route.request().postDataJSON(); commands.push({ provider: 'spotify', ...command });
      const p = spotify.playback;
      if (delayedSpotify) {
        const before = structuredClone(spotify);
        if (command.action === 'seek') {
          setTimeout(() => { p.positionMs = command.value; }, 900);
          await new Promise(resolve => setTimeout(resolve, 600));
        } else if (command.action === 'next') {
          setTimeout(() => { p.track.title = 'Settled next track'; p.track.album = 'Settled next album'; p.positionMs = 0; }, 250);
        } else throw new Error('Unexpected delayed command');
        return route.fulfill({ json: before });
      }
      switch (command.action) {
        case 'pause': p.playing = false; break;
        case 'play': p.playing = true; break;
        case 'next': p.track.title = 'Next live track'; p.track.album = 'Next album'; p.track.artworkUrl = nextArtwork; p.positionMs = 0; break;
        case 'previous': p.track.title = 'Live track'; p.track.album = 'Live album'; p.track.artworkUrl = firstArtwork; p.positionMs = 0; break;
        case 'seek': p.positionMs = command.value; break;
        case 'volume': p.device.volume = command.value; break;
        case 'shuffle': p.shuffle = command.value; break;
        case 'repeat': p.repeat = command.value; break;
      }
    }
    if (failTransport) return route.fulfill({ status: 503, json: {} });
    await route.fulfill({ json: spotify });
  });
  function ok(name, condition) { assert.ok(condition, name); checks.push(name); console.log('PASS', name); }
  async function open(view) {
    if (await page.locator('.task-view').isVisible()) await page.locator('.task-view>.close-task').click();
    await page.locator('.control-trigger').click();
    await page.locator(`.control-panel [data-open="${view}"]`).click();
  }
  async function range(kind, value) {
    await page.locator(`[data-input="${kind}"]`).evaluate((element, value) => {
      element.value = String(value); element.dispatchEvent(new Event('input', { bubbles: true })); element.dispatchEvent(new Event('change', { bubbles: true }));
    }, value);
  }
  async function artworkReady(selector, url) {
    await page.waitForFunction(({ selector, url }) => {
      const img = document.querySelector(selector);
      return img?.getAttribute('src') === url && img.complete && img.naturalWidth > 0;
    }, { selector, url });
  }
  try {
    await page.goto(base + '/?fullscreen=0');
    await page.waitForFunction(() => document.querySelector('[data-peek-song]')?.textContent === 'Live track');
    ok('home displays provider track and converted temperature', await page.locator('[data-home-temperature]').textContent() === '25');
    await artworkReady('.album-icon img', firstArtwork);
    ok('compact music player displays the Spotify cover', await page.locator('.album-icon img').isVisible());
    await page.screenshot({ path: out + '/live-home-800.png' });
    await open('lights');
    ok('allowlisted light replaces demo lights', await page.locator('[data-light-row]').count() === 1 && (await page.locator('.light-name strong').textContent()) === 'Ceiling light');
    ok('off state and observed brightness are shown', await page.locator('[data-row-brightness]').textContent() === 'Off' && await page.locator('[data-brightness]').textContent() === '50%');
    await page.locator('[data-light-toggle="0"]').click();
    await page.waitForFunction(() => document.querySelector('[data-light-toggle="0"]')?.getAttribute('aria-pressed') === 'true');
    ok('touch toggles the configured entity', commands.at(-1).entityId === 'light.hue_color_lamp_1' && commands.at(-1).action === 'on');
    await range('brightness', 42);
    await page.waitForFunction(() => document.querySelector('[data-brightness]')?.textContent === '42%');
    await page.locator('[data-color="Blue"]').click();
    await page.waitForFunction(() => document.querySelector('[data-color="Blue"]')?.getAttribute('aria-pressed') === 'true');
    ok('Hue hs lights support RGB presets', commands.at(-1).action === 'color' && commands.at(-1).rgb[2] === 255);
    await page.locator('[data-color="Warm white"]').click();
    await page.waitForFunction(() => document.querySelector('[data-color="Warm white"]')?.getAttribute('aria-pressed') === 'true');
    ok('white presets send Kelvin', commands.at(-1).action === 'temperature' && commands.at(-1).value === 2700);
    await page.screenshot({ path: out + '/live-lights-800.png' });
    failLight = true;
    await page.locator('[data-light-toggle="0"]').click();
    await page.waitForFunction(() => document.querySelector('.toast')?.textContent.includes('Light request failed'));
    ok('rejected actions retain observed state', await page.locator('[data-light-toggle="0"]').getAttribute('aria-pressed') === 'true');
    failLight = false;
    await page.locator('[data-action="all-off"]').click();
    await page.waitForFunction(() => document.querySelector('[data-row-brightness]')?.textContent === 'Off');
    await open('music');
    await artworkReady('.record img', firstArtwork);
    ok('expanded music view displays the same cover without cropping', await page.locator('.record img').evaluate(img => getComputedStyle(img).objectFit === 'contain'));
    ok('music view shows the Spotify album beneath the artist', await page.locator('[data-album-title]').textContent() === 'Album · Live album');
    await page.locator('.play-control').click();
    await page.waitForFunction(() => document.querySelector('.play-control')?.getAttribute('aria-label') === 'Play');
    ok('Spotify pause comes from provider state', commands.at(-1).action === 'pause');
    await page.locator('[data-action="next"]').click();
    await page.waitForFunction(() => document.querySelector('.music-info h2')?.textContent === 'Next live track');
    ok('album follows a track change', await page.locator('[data-album-title]').textContent() === 'Album · Next album');
    await artworkReady('.record img', nextArtwork); await artworkReady('.album-icon img', nextArtwork);
    ok('track changes update artwork in both players', true);
    await page.locator('[data-action="previous"]').click();
    await page.waitForFunction(() => document.querySelector('.music-info h2')?.textContent === 'Live track');
    await range('seek', 70);
    await page.waitForFunction(() => document.querySelector('[data-track-time]')?.textContent === '01:10');
    ok('seek uses milliseconds', commands.at(-1).action === 'seek' && commands.at(-1).value === 70000);
    delayedSpotify = true;
    const seekStarted = Date.now();
    await range('seek', 120); await page.waitForTimeout(350);
    ok('scrubbing stays at the requested position while the command is pending', await page.locator('[data-track-time]').textContent() === '02:00');
    await page.waitForTimeout(350);
    ok('a stale command snapshot does not snap the seek position back', await page.locator('[data-track-time]').textContent() === '02:00' && await page.locator('.provider-note').textContent().then(text => text.includes('Seeking')));
    await page.waitForFunction(() => !document.querySelector('.provider-note')?.textContent.includes('Seeking'));
    ok('seek refreshes promptly after Spotify catches up', Date.now() - seekStarted < 2500);
    const nextStarted = Date.now();
    await page.locator('[data-action="next"]').click();
    await page.waitForFunction(() => document.querySelector('.music-info h2')?.textContent === 'Settled next track');
    ok('next refreshes without the old five-second delay', Date.now() - nextStarted < 2000 && await page.locator('[data-album-title]').textContent() === 'Album · Settled next album');
    delayedSpotify = false;
    await range('volume', 0);
    await page.waitForFunction(() => document.querySelector('[data-input="volume"]')?.value === '0' && !document.querySelector('[data-input="volume"]')?.disabled);
    ok('volume preserves zero', commands.at(-1).action === 'volume' && commands.at(-1).value === 0);
    await page.getByRole('button', { name: 'Unmute Spotify' }).click();
    await page.waitForFunction(() => document.querySelector('[data-input="volume"]')?.value === '30' && !document.querySelector('[data-input="volume"]')?.disabled);
    await page.getByRole('button', { name: 'Mute Spotify' }).click();
    await page.waitForFunction(() => document.querySelector('[data-input="volume"]')?.value === '0' && !document.querySelector('[data-input="volume"]')?.disabled);
    ok('Spotify mute restores the last observed nonzero volume', commands.at(-2).value === 30 && commands.at(-1).value === 0);

    await page.locator('[data-action="shuffle"]').click();
    await page.waitForFunction(() => document.querySelector('[data-action="shuffle"]')?.getAttribute('aria-pressed') === 'true');
    await page.locator('[data-action="repeat"]').click();
    await page.waitForFunction(() => document.querySelector('[data-action="repeat"]')?.getAttribute('aria-label') === 'Repeat: context');
    await page.screenshot({ path: out + '/live-music-800.png' });
    const beforeSimulator = commands.length;
    await page.evaluate(() => document.querySelector('main').dispatchEvent(new CustomEvent('vcm-intent', { detail: { intent: 'media.play', parameters: {} } })));
    await page.waitForTimeout(700);
    ok('simulator cannot operate live providers', commands.length === beforeSimulator && await page.locator('.play-control').getAttribute('aria-label') === 'Play');
    const box = await page.locator('.music-info').boundingBox();
    ok('live music fits 800 by 480', box.x >= 0 && box.y >= 0 && box.x + box.width <= 800 && box.y + box.height <= 480);
    await page.locator('.task-view>.close-task').click();
    await page.locator('.demo-launch').click();
    ok('Play music is labeled as a live shortcut', await page.locator('[data-demo="6"] .shortcut-live').textContent() === 'Live');
    await page.locator('[data-demo="6"]').click();
    await page.waitForFunction(() => document.querySelector('.play-control')?.getAttribute('aria-label') === 'Pause');
    ok('Play music resumes Spotify without a song or search text', JSON.stringify(commands.at(-1)) === JSON.stringify({ provider: 'spotify', action: 'play' }));
    await page.setViewportSize({ width: 1024, height: 600 });
    await page.screenshot({ path: out + '/live-music-1024.png' });
    spotify.playback.track.artworkUrl = null;
    await page.waitForFunction(() => !document.querySelector('.record img') && !document.querySelector('.album-icon img'));
    ok('missing artwork restores both placeholders without clearing the track', await page.locator('.record svg').count() === 1 && await page.locator('.album-icon .mini-equalizer').count() === 1 && await page.locator('.music-info h2').textContent() === 'Settled next track');
    spotify.playback.track.artworkUrl = 'https://i.scdn.co/image/qa-cover-broken';
    const brokenImage = page.waitForResponse('https://i.scdn.co/image/qa-cover-broken');
    await brokenImage;
    await page.waitForFunction(() => !document.querySelector('.record img') && !document.querySelector('.album-icon img'));
    ok('failed artwork loads fall back without breaking playback controls', await page.locator('.record svg').count() === 1 && await page.locator('.album-icon .mini-equalizer').count() === 1 && !await page.locator('.play-control').isDisabled());
    spotify.playback.track.artworkUrl = nextArtwork;
    await artworkReady('.record img', nextArtwork); await artworkReady('.album-icon img', nextArtwork);
    ok('valid artwork recovers after a failed cover', true);
    await open('lights');
    await page.screenshot({ path: out + '/live-lights-1024.png' });
    ha.entities = []; await page.waitForFunction(() => document.querySelectorAll('[data-light-row]').length === 0);
    ok('entity removal renders empty state without crashes', await page.locator('.light-list').textContent().then(text => text.includes('No lights available')));
    spotify.status = 'not_authorized'; spotify.playback = null;
    await page.waitForTimeout(5100); await open('music');
    ok('unauthorized Spotify shows connect action and disables playback', await page.getByRole('button', { name: 'Connect Spotify' }).isVisible() && await page.locator('.play-control').isDisabled());
    ok('unavailable playback clears the previous album', await page.locator('[data-album-title]').count() === 0);
    ok('unavailable playback clears the previous cover in both players', await page.locator('.record img').count() === 0 && await page.locator('.album-icon img').count() === 0);
    const beforeConnect = commands.length;
    await page.locator('.task-view>.close-task').click();
    await page.locator('.demo-launch').click(); await page.locator('[data-demo="6"]').click();
    ok('unauthorized Play music opens the connection prompt', await page.getByRole('button', { name: 'Connect Spotify' }).isVisible() && commands.length === beforeConnect);
    await page.waitForTimeout(500);
    await page.screenshot({ path: out + '/spotify-connect.png' });
    failTransport = true; await page.waitForTimeout(5200);
    ok('transport failure never restores demo music', await page.locator('.music-info h2').textContent() === 'Spotify' && await page.locator('.play-control').isDisabled());
    failTransport = false;
    ha.enabled = false; ha.status = 'disabled'; spotify.enabled = false; spotify.status = 'disabled';
    await page.reload(); await open('lights');
    await page.waitForFunction(() => document.querySelectorAll('[data-light-row]').length === 4);
    ok('disabled integration retains interactive demo lights', !await page.locator('[data-light-toggle="0"]').isDisabled());
    ok('no browser runtime errors', errors.length === 0);
    fs.writeFileSync(out + '/checks.json', JSON.stringify({ checks, errors, commands, environment: 'Chromium with controlled provider responses; no real device actions' }, null, 2));
    console.log('FINISHED', checks.length);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

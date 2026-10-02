const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{}),args:['--enable-unsafe-swiftshader']});
 const page=await browser.newPage({viewport:{width:800,height:480},hasTouch:true}),checks=[],errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 const ok=(name,value)=>{assert.ok(value,name);checks.push(name);console.log('PASS',name)};
 await page.clock.install();
 const base=process.env.TEST_BASE_URL||'http://127.0.0.1:3000/?fullscreen=0';
 async function reset(){await page.goto(base);await page.waitForTimeout(100);}
 async function intent(intent){await page.evaluate(intent=>document.querySelector('main').dispatchEvent(new CustomEvent('vcm-intent',{detail:{intent,parameters:intent==='light.brightness'?{entity_id:'light.living_room',brightness_pct:40}:{}}})),intent);await page.waitForTimeout(650);}
 try {
 if(!process.env.SKIP_TOAST){await reset();await intent('light.brightness');ok('first light brightness intent displays a toast without runtime failure',errors.length===0&&await page.locator('.toast').isVisible());}
 await reset();await intent('demo.reminder_near');await page.clock.fastForward(46000);await page.locator('[data-action=alert-snooze]').click();ok('snoozing a near-deadline reminder discards its stale queued due alert',!(await page.locator('.alert-surface').isVisible()));
 await page.clock.fastForward(301000);ok('snoozed reminder rings again five minutes later',await page.locator('.alert-surface').isVisible());
 await page.locator('[data-action=alert-dismiss]').click();
 await reset();await intent('demo.timer_due');ok('alert perimeter has rounded corners on every edge',await page.locator('.attention-edge').evaluate(e=>{const s=getComputedStyle(e);return s.borderRadius==='24px'&&s.top==='3px'&&s.bottom==='3px'&&s.left==='3px'&&s.right==='3px'&&s.boxShadow.includes('inset')}));
 ok('flash rhythm is 3.6 seconds with static edge geometry',await page.locator('.attention-edge').evaluate(e=>getComputedStyle(e).animationDuration==='3.6s'));
 await page.clock.fastForward(299000);ok('timer alert remains visible before five minutes',await page.locator('.alert-surface').isVisible());await page.clock.fastForward(1500);ok('timer alert closes after five minutes',!(await page.locator('.alert-surface').isVisible()));
 await intent('demo.alarm_due');await intent('demo.timer_due');await page.clock.fastForward(299000);ok('first queued alarm gets its full lifetime',(await page.locator('.alert-content .subtle').textContent())==='Alarm');await page.clock.fastForward(1500);ok('next queued timer opens after alarm timeout',(await page.locator('.alert-content h2').textContent())==='Timer finished');await page.clock.fastForward(299000);ok('queued timer receives a fresh five-minute lifetime',await page.locator('.alert-surface').isVisible());await page.clock.fastForward(1500);ok('queued timer also closes',!(await page.locator('.alert-surface').isVisible()));
 await reset();await intent('demo.reminder_due');await page.clock.fastForward(301000);ok('reminder alert closes without completing reminder',!(await page.locator('.alert-surface').isVisible())&&(await page.locator('[data-reminder-count]').textContent())==='3');
 await intent('demo.alarm_due');await page.evaluate(()=>Object.defineProperty(document,'hidden',{configurable:true,value:true}));await page.clock.fastForward(301000);ok('hidden display pauses alert rendering work',await page.locator('.alert-surface').isVisible());await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,value:false});document.dispatchEvent(new Event('visibilitychange'));});ok('returning after five minutes clears expired alert',!(await page.locator('.alert-surface').isVisible()));
 await page.emulateMedia({reducedMotion:'reduce'});await intent('demo.timer_due');ok('reduced motion is steady and still visible',await page.locator('.attention-edge').evaluate(e=>getComputedStyle(e).animationName==='none'&&Number(getComputedStyle(e).opacity)>0));
 ok('no browser errors',errors.length===0);
 } finally {fs.writeFileSync(process.env.TEST_REPORT||'docs/testing/ambient/alerts.json',JSON.stringify({base,checks,errors},null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});

# PC-v1 voice command demonstration suite

Status: **NOT RUN — no trained PC-v1 ONNX bundle or original demo audio is available.** This file lists expected outputs, not observed model results. The implementation is not ready for meaningful voice acceptance testing until a trained, development-selected and calibrated PC-v1 bundle exists. Physical production qualification remains separate.

The full [machine-readable suite](../experiments/phrase-contract-v1/reports/demo-suite/commands.json) contains all 570 positive phrases and 200 rejection prompts from the fixed contract. Every actual output is null. The subset below covers every supported intent across all ten use cases, plus one rejection per use case. No phrase or output schema was changed. Source approval metadata is preserved; this document does not create an approval receipt.

## What ONNX outputs

The graph returns `ctc` scores over 39 phones plus blank and six `next_state_*` cache tensors. It does not directly return JSON. The deployed frontend, finite command decoder, endpoint detector and calibrated acceptance policy produce the JSON shown below. No transcript or external action executor is involved.

## Running the demonstration after a real candidate exists

Use the existing `stage-candidate` command in the [execution ledger](../experiments/phrase-contract-v1/PROGRESS.md) to produce an unqualified test directory from the selected calibrated bundle. Do not enable a disabled policy by hand. A missing bundle is a blocked precondition, never a passing rejection.

From that directory, with the appropriate environment installed:

```powershell
python pcv1.py --check
python pcv1.py --wav PATH_TO_ORIGINAL_MONO_16000_HZ_RECORDING.wav
python pcv1.py --list-mics
python pcv1.py --mic ACTUAL_DEVICE_INDEX
```

The microphone command arms one utterance when Enter is pressed. Use the listed device index. Record the literal prompt, preserve natural speech and trailing silence for the endpoint, and do not splice or synthesize evaluation speech. Keep demonstration recordings separate from the sealed confirmation set. Save actual output, recording and PCM hashes, speaker/session/condition, bundle binding and latency in a new run receipt; do not overwrite the expected suite.

For a positive case, compare the complete final JSON to the expected output including typed parameter values. For a rejection, require a healthy completed endpoint, `accepted=false`, `command=null`, and `reason=unknown`. `policy_disabled`, `invalid_bundle`, `capture_fault`, `tentative`, a missing bundle, or a crashed run are blocked/invalid trials, not successful rejections. A raw top-ranked candidate is not acceptance.

Passing this small demonstration does not replace the complete prescribed acoustic and physical qualification gates. The system recognizes the finite request; it does not play music, answer live weather, operate devices, or send calls/messages.

## Demonstration cases

### UC-01: Music

| Test ID | Say exactly | Expected final pipeline output | Actual |
| --- | --- | --- | --- |
| UC01-PL-FOCUS-A | play focus playlist | `{"accepted":true,"command":{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.focus"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC01-TR-RIVER-ALPHA | play river by alpha | `{"accepted":true,"command":{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.river"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC01-REJ-01 | do not play river by alpha | `{"accepted":false,"command":null,"reason":"unknown","schema_version":2}` | NOT_RUN |

### UC-02: Questions

| Test ID | Say exactly | Expected final pipeline output | Actual |
| --- | --- | --- | --- |
| UC02-TIME-A | what time is it | `{"accepted":true,"command":{"intent":"TIME_QUERY","timezone":"Asia/Manila"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC02-DATE-A | what is the date | `{"accepted":true,"command":{"intent":"DATE_QUERY","timezone":"Asia/Manila"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC02-WEATHER-A | what is the weather | `{"accepted":true,"command":{"intent":"WEATHER_QUERY","location_id":"location.home","period":"today"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC02-REJ-01 | do not tell me the time | `{"accepted":false,"command":null,"reason":"unknown","schema_version":2}` | NOT_RUN |

### UC-03: Light power

| Test ID | Say exactly | Expected final pipeline output | Actual |
| --- | --- | --- | --- |
| UC03-ON-K-A | turn on kitchen lamp | `{"accepted":true,"command":{"entity_id":"light.kitchen","intent":"POWER_ON"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC03-OFF-K-A | turn off kitchen lamp | `{"accepted":true,"command":{"entity_id":"light.kitchen","intent":"POWER_OFF"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC03-REJ-01 | do not turn on kitchen lamp | `{"accepted":false,"command":null,"reason":"unknown","schema_version":2}` | NOT_RUN |

### UC-04: Light brightness/color

| Test ID | Say exactly | Expected final pipeline output | Actual |
| --- | --- | --- | --- |
| UC04-BRI-K-037-A | set kitchen lamp to thirty seven percent | `{"accepted":true,"command":{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":37},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC04-DELTA-K-P05 | increase kitchen lamp by five percent | `{"accepted":true,"command":{"delta":5,"entity_id":"light.kitchen","intent":"BRIGHTNESS_DELTA"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC04-COLOR-K-RED | set kitchen lamp to red | `{"accepted":true,"command":{"color":"red","entity_id":"light.kitchen","intent":"COLOR_SET"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC04-REJ-01 | do not set kitchen lamp to seventy three percent | `{"accepted":false,"command":null,"reason":"unknown","schema_version":2}` | NOT_RUN |

### UC-05: Timers

| Test ID | Say exactly | Expected final pipeline output | Actual |
| --- | --- | --- | --- |
| UC05-SET-0060-A | set a timer for one minute | `{"accepted":true,"command":{"duration_seconds":60,"intent":"TIMER_SET","timer_id":"timer.primary"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC05-CANCEL-A | cancel timer | `{"accepted":true,"command":{"intent":"TIMER_CANCEL","timer_id":"timer.primary"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC05-REJ-01 | do not set a timer for five minutes | `{"accepted":false,"command":null,"reason":"unknown","schema_version":2}` | NOT_RUN |

### UC-06: Alarms

| Test ID | Say exactly | Expected final pipeline output | Actual |
| --- | --- | --- | --- |
| UC06-SET-0600-A | set an alarm for six a m | `{"accepted":true,"command":{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:00","timezone":"Asia/Manila"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC06-CANCEL-A | cancel alarm | `{"accepted":true,"command":{"alarm_id":"alarm.primary","intent":"ALARM_CANCEL"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC06-REJ-01 | do not set an alarm for seven a m | `{"accepted":false,"command":null,"reason":"unknown","schema_version":2}` | NOT_RUN |

### UC-07: Thermostat

| Test ID | Say exactly | Expected final pipeline output | Actual |
| --- | --- | --- | --- |
| UC07-ABS-16 | set thermostat to sixteen degrees celsius | `{"accepted":true,"command":{"celsius":16,"entity_id":"climate.home","intent":"TEMPERATURE_SET"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC07-DELTA-P1 | increase thermostat by one degree celsius | `{"accepted":true,"command":{"delta_celsius":1,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC07-REJ-01 | do not set thermostat to twenty degrees celsius | `{"accepted":false,"command":null,"reason":"unknown","schema_version":2}` | NOT_RUN |

### UC-08: Media controls

| Test ID | Say exactly | Expected final pipeline output | Actual |
| --- | --- | --- | --- |
| UC08-PAUSE-A | pause music | `{"accepted":true,"command":{"entity_id":"media.main","intent":"MEDIA_PAUSE"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC08-RESUME-A | resume music | `{"accepted":true,"command":{"entity_id":"media.main","intent":"MEDIA_RESUME"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC08-STOP-A | stop music | `{"accepted":true,"command":{"entity_id":"media.main","intent":"MEDIA_STOP"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC08-NEXT-A | next track | `{"accepted":true,"command":{"entity_id":"media.main","intent":"MEDIA_NEXT"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC08-PREVIOUS-A | previous track | `{"accepted":true,"command":{"entity_id":"media.main","intent":"MEDIA_PREVIOUS"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC08-MUTE-A | mute audio | `{"accepted":true,"command":{"entity_id":"media.main","intent":"MEDIA_MUTE"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC08-UNMUTE-A | unmute audio | `{"accepted":true,"command":{"entity_id":"media.main","intent":"MEDIA_UNMUTE"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC08-VOL-ABS-050-A | set volume to fifty percent | `{"accepted":true,"command":{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":50},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC08-VOL-DELTA-P05 | increase volume by five percent | `{"accepted":true,"command":{"delta":5,"entity_id":"media.main","intent":"VOLUME_DELTA"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC08-REJ-01 | do not pause music | `{"accepted":false,"command":null,"reason":"unknown","schema_version":2}` | NOT_RUN |

### UC-09: Reminders/lists

| Test ID | Say exactly | Expected final pipeline output | Actual |
| --- | --- | --- | --- |
| UC09-REM-WATER-A | remind me to drink water in ten minutes | `{"accepted":true,"command":{"content_id":"reminder_content.water","delay_seconds":600,"intent":"REMINDER_SET","reminder_id":"reminder.water"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC09-REM-CANCEL-WATER | cancel my water reminder | `{"accepted":true,"command":{"intent":"REMINDER_CANCEL","reminder_id":"reminder.water"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC09-LIST-ADD-MILK-A | add milk to my shopping list | `{"accepted":true,"command":{"intent":"LIST_ADD","item_id":"item.milk","list_id":"list.shopping"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC09-LIST-REMOVE-MILK | remove milk from my shopping list | `{"accepted":true,"command":{"intent":"LIST_REMOVE","item_id":"item.milk","list_id":"list.shopping"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC09-LIST-READ-A | read my shopping list | `{"accepted":true,"command":{"intent":"LIST_READ","list_id":"list.shopping"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC09-REJ-01 | do not remind me to drink water in ten minutes | `{"accepted":false,"command":null,"reason":"unknown","schema_version":2}` | NOT_RUN |

### UC-10: Calls/messages

| Test ID | Say exactly | Expected final pipeline output | Actual |
| --- | --- | --- | --- |
| UC10-CALL-ALEX-A | call alex kim | `{"accepted":true,"command":{"contact_id":"contact.alex_kim","intent":"CALL_START"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC10-MSG-ALEX-WAY-A | message alex kim i am on my way | `{"accepted":true,"command":{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.on_my_way"},"reason":"accepted","schema_version":2}` | NOT_RUN |
| UC10-REJ-01 | do not call alex kim | `{"accepted":false,"command":null,"reason":"unknown","schema_version":2}` | NOT_RUN |

## Exact blocker

The required original-human training/development/calibration/confirmation recordings and their frozen provenance/speaker manifests are absent. The existing 48,800-slot plan contains zero obtained recordings. No trained checkpoint, fitted normalization, selected endpoint or calibrated PC-v1 policy is present. Therefore there is no legitimate ONNX speech result to fill into this suite yet. The previous engineering tests and temporary random-weight export do not meet this requirement.

Contract SHA-256: `ae004037a95bf8ab8fb170896927ab262898f4d11e1e5a251fdff0bafa10c35f`.

See the [execution ledger](../experiments/phrase-contract-v1/PROGRESS.md) for the fixed acquisition, training, export, calibration and qualification sequence.

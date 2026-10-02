# PCV demonstration commands

35 commands, selected before model fitting and final test. Expectations only: actual recognition is not measured yet.
Say Hey Jarvis, wait for command-ready, then use the exact wording. The VCM emits intent and typed parameters; it does not emit a transcript. Word cues below are explanatory expectations, not an extra transcription output.

Stable phrase IDs, intent labels, typed parameters and keyword cues are in the [frozen machine-readable demo](../experiments/pcv-delivery-v1/demo.json). The [full two-phase guide](PCV-SYNTHETIC-V2-VERBATIMS.md) contains all 2,768 study verbatims.

| Use case | Exact verbatim | Reference word cues | Expected intent and parameters |
| --- | --- | --- | --- |
| UC-01 | play butter | play, butter | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.bts_butter"}` |
| UC-01 | play dynamite by BTS | play, dynamite, by, bts | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.bts_dynamite"}` |
| UC-01 | play stay by BTS | play, stay, by, bts | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.bts_stay"}` |
| UC-01 | play stay by Blackpink | play, stay, by, blackpink | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.blackpink_stay"}` |
| UC-02 | what time is it | what, time, is, it | `{"intent":"TIME_QUERY","timezone":"Asia/Manila"}` |
| UC-02 | what is the date | what, is, the, date | `{"intent":"DATE_QUERY","timezone":"Asia/Manila"}` |
| UC-02 | what is the weather today | what, is, the, weather, today | `{"intent":"WEATHER_QUERY","location_id":"location.home","period":"today"}` |
| UC-03 | turn on kitchen lamp | turn, on, kitchen, lamp | `{"entity_id":"light.kitchen","intent":"POWER_ON"}` |
| UC-03 | turn off kitchen lamp | turn, off, kitchen, lamp | `{"entity_id":"light.kitchen","intent":"POWER_OFF"}` |
| UC-03 | turn on bedroom lamp | turn, on, bedroom, lamp | `{"entity_id":"light.bedroom","intent":"POWER_ON"}` |
| UC-04 | set kitchen lamp to zero percent | set, kitchen, lamp, to, zero, percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":0}` |
| UC-04 | set kitchen lamp to one hundred percent | set, kitchen, lamp, to, one, hundred, percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":100}` |
| UC-04 | set kitchen lamp to thirty seven percent | set, kitchen, lamp, to, thirty, seven, percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":37}` |
| UC-04 | set kitchen lamp to cyan | set, kitchen, lamp, to, cyan | `{"color":"cyan","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC-04 | set kitchen lamp to red | set, kitchen, lamp, to, red | `{"color":"red","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC-05 | set a timer for one minute | set, a, timer, for, one, minute | `{"duration_seconds":60,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC-05 | set a timer for five minutes | set, a, timer, for, five, minutes | `{"duration_seconds":300,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC-05 | set a timer for two hours | set, a, timer, for, two, hours | `{"duration_seconds":7200,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC-06 | set an alarm for six thirty a m | set, an, alarm, for, six, thirty, a, m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:30","timezone":"Asia/Manila"}` |
| UC-06 | set an alarm for six p m | set, an, alarm, for, six, p, m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"18:00","timezone":"Asia/Manila"}` |
| UC-06 | cancel alarm | cancel, alarm | `{"alarm_id":"alarm.primary","intent":"ALARM_CANCEL"}` |
| UC-07 | set thermostat to twenty one degrees celsius | set, thermostat, to, twenty, one, degrees, celsius | `{"celsius":21,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC-07 | increase thermostat by two degrees celsius | increase, thermostat, by, two, degrees, celsius | `{"delta_celsius":2,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC-07 | decrease thermostat by one degree celsius | decrease, thermostat, by, one, degree, celsius | `{"delta_celsius":-1,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC-08 | pause music | pause, music | `{"entity_id":"media.main","intent":"MEDIA_PAUSE"}` |
| UC-08 | resume music | resume, music | `{"entity_id":"media.main","intent":"MEDIA_RESUME"}` |
| UC-08 | next track | next, track | `{"entity_id":"media.main","intent":"MEDIA_NEXT"}` |
| UC-08 | set volume to zero percent | set, volume, to, zero, percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":0}` |
| UC-08 | set volume to one hundred percent | set, volume, to, one, hundred, percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":100}` |
| UC-09 | set a reminder for one minute | set, a, reminder, for, one, minute | `{"delay_seconds":60,"intent":"REMINDER_SET","reminder_id":"reminder.timer"}` |
| UC-09 | set a reminder for two hours | set, a, reminder, for, two, hours | `{"delay_seconds":7200,"intent":"REMINDER_SET","reminder_id":"reminder.timer"}` |
| UC-09 | set a reminder at twelve p m | set, a, reminder, at, twelve, p, m | `{"intent":"REMINDER_SET","reminder_id":"reminder.timer","time_local":"12:00","timezone":"Asia/Manila"}` |
| UC-10 | call alex kim | call, alex, kim | `{"contact_id":"contact.alex_kim","intent":"CALL_START"}` |
| UC-10 | call jamie cruz | call, jamie, cruz | `{"contact_id":"contact.jamie_cruz","intent":"CALL_START"}` |
| UC-10 | message alex kim i am on my way | message, alex, kim, i, am, on, my, way | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |

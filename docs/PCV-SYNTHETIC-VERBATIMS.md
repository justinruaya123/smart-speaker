# Synthetic benchmark: full test verbatims

Expected outputs only; generated audio and measured results are recorded separately.

| ID | Phase | Use case | Say exactly | Expected command (null means reject) |
| --- | --- | --- | --- | --- |
| UC01-PL-FOCUS-A | 1 | UC-01 | play focus playlist | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.focus"}` |
| UC01-PL-FOCUS-A-N1 | 2 | UC-01 | could you play focus playlist, please | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.focus"}` |
| UC01-PL-FOCUS-A-N2 | 2 | UC-01 | I would like you to play focus playlist for me | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.focus"}` |
| UC01-PL-FOCUS-B | 1 | UC-01 | start focus playlist | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.focus"}` |
| UC01-PL-FOCUS-B-N1 | 2 | UC-01 | could you start focus playlist, please | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.focus"}` |
| UC01-PL-FOCUS-B-N2 | 2 | UC-01 | I would like you to start focus playlist for me | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.focus"}` |
| UC01-PL-FAVORITES-A | 1 | UC-01 | play favorites playlist | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.favorites"}` |
| UC01-PL-FAVORITES-A-N1 | 2 | UC-01 | could you play favorites playlist, please | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.favorites"}` |
| UC01-PL-FAVORITES-A-N2 | 2 | UC-01 | I would like you to play favorites playlist for me | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.favorites"}` |
| UC01-PL-FAVORITES-B | 1 | UC-01 | start favorites playlist | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.favorites"}` |
| UC01-PL-FAVORITES-B-N1 | 2 | UC-01 | could you start favorites playlist, please | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.favorites"}` |
| UC01-PL-FAVORITES-B-N2 | 2 | UC-01 | I would like you to start favorites playlist for me | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.favorites"}` |
| UC01-PL-WORKOUT-A | 1 | UC-01 | play workout playlist | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.workout"}` |
| UC01-PL-WORKOUT-A-N1 | 2 | UC-01 | could you play workout playlist, please | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.workout"}` |
| UC01-PL-WORKOUT-A-N2 | 2 | UC-01 | I would like you to play workout playlist for me | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.workout"}` |
| UC01-PL-WORKOUT-B | 1 | UC-01 | start workout playlist | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.workout"}` |
| UC01-PL-WORKOUT-B-N1 | 2 | UC-01 | could you start workout playlist, please | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.workout"}` |
| UC01-PL-WORKOUT-B-N2 | 2 | UC-01 | I would like you to start workout playlist for me | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.workout"}` |
| UC01-GENERIC-A | 1 | UC-01 | play music | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.favorites"}` |
| UC01-GENERIC-A-N1 | 2 | UC-01 | could you play music, please | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.favorites"}` |
| UC01-GENERIC-A-N2 | 2 | UC-01 | I would like you to play music for me | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.favorites"}` |
| UC01-GENERIC-B | 1 | UC-01 | start music | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.favorites"}` |
| UC01-GENERIC-B-N1 | 2 | UC-01 | could you start music, please | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.favorites"}` |
| UC01-GENERIC-B-N2 | 2 | UC-01 | I would like you to start music for me | `{"entity_id":"media.main","intent":"MUSIC_PLAYLIST","playlist_id":"playlist.favorites"}` |
| UC01-TR-RIVER-ALPHA | 1 | UC-01 | play river by alpha | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.river"}` |
| UC01-TR-RIVER-ALPHA-N1 | 2 | UC-01 | could you play river by alpha, please | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.river"}` |
| UC01-TR-RIVER-ALPHA-N2 | 2 | UC-01 | I would like you to play river by alpha for me | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.river"}` |
| UC01-TR-RIVER-BETA | 1 | UC-01 | play river by beta | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.river.beta"}` |
| UC01-TR-RIVER-BETA-N1 | 2 | UC-01 | could you play river by beta, please | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.river.beta"}` |
| UC01-TR-RIVER-BETA-N2 | 2 | UC-01 | I would like you to play river by beta for me | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.river.beta"}` |
| UC01-TR-73-BARE | 1 | UC-01 | play seventy three | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.seventy_three"}` |
| UC01-TR-73-BARE-N1 | 2 | UC-01 | could you play seventy three, please | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.seventy_three"}` |
| UC01-TR-73-BARE-N2 | 2 | UC-01 | I would like you to play seventy three for me | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.seventy_three"}` |
| UC01-TR-73-ALPHA | 1 | UC-01 | play seventy three by alpha | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.seventy_three"}` |
| UC01-TR-73-ALPHA-N1 | 2 | UC-01 | could you play seventy three by alpha, please | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.seventy_three"}` |
| UC01-TR-73-ALPHA-N2 | 2 | UC-01 | I would like you to play seventy three by alpha for me | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.seventy_three"}` |
| UC01-TR-73-BETA | 1 | UC-01 | play seventy three by beta | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.seventy_three"}` |
| UC01-TR-73-BETA-N1 | 2 | UC-01 | could you play seventy three by beta, please | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.seventy_three"}` |
| UC01-TR-73-BETA-N2 | 2 | UC-01 | I would like you to play seventy three by beta for me | `{"entity_id":"media.main","intent":"MEDIA_PLAY","track_id":"track.seventy_three"}` |
| UC02-TIME-A | 1 | UC-02 | what time is it | `{"intent":"TIME_QUERY","timezone":"Asia/Manila"}` |
| UC02-TIME-A-N1 | 2 | UC-02 | can you tell me what time it is, please | `{"intent":"TIME_QUERY","timezone":"Asia/Manila"}` |
| UC02-TIME-A-N2 | 2 | UC-02 | I would like to know what time it is | `{"intent":"TIME_QUERY","timezone":"Asia/Manila"}` |
| UC02-TIME-B | 1 | UC-02 | tell me the time | `{"intent":"TIME_QUERY","timezone":"Asia/Manila"}` |
| UC02-TIME-B-N1 | 2 | UC-02 | could you tell me the time, please | `{"intent":"TIME_QUERY","timezone":"Asia/Manila"}` |
| UC02-TIME-B-N2 | 2 | UC-02 | I would like you to tell me the time for me | `{"intent":"TIME_QUERY","timezone":"Asia/Manila"}` |
| UC02-DATE-A | 1 | UC-02 | what is the date | `{"intent":"DATE_QUERY","timezone":"Asia/Manila"}` |
| UC02-DATE-A-N1 | 2 | UC-02 | can you tell me the date, please | `{"intent":"DATE_QUERY","timezone":"Asia/Manila"}` |
| UC02-DATE-A-N2 | 2 | UC-02 | I would like to know the date | `{"intent":"DATE_QUERY","timezone":"Asia/Manila"}` |
| UC02-DATE-B | 1 | UC-02 | tell me today's date | `{"intent":"DATE_QUERY","timezone":"Asia/Manila"}` |
| UC02-DATE-B-N1 | 2 | UC-02 | could you tell me today's date, please | `{"intent":"DATE_QUERY","timezone":"Asia/Manila"}` |
| UC02-DATE-B-N2 | 2 | UC-02 | I would like you to tell me today's date for me | `{"intent":"DATE_QUERY","timezone":"Asia/Manila"}` |
| UC02-WEATHER-A | 1 | UC-02 | what is the weather | `{"intent":"WEATHER_QUERY","location_id":"location.home","period":"today"}` |
| UC02-WEATHER-A-N1 | 2 | UC-02 | can you tell me the weather, please | `{"intent":"WEATHER_QUERY","location_id":"location.home","period":"today"}` |
| UC02-WEATHER-A-N2 | 2 | UC-02 | I would like to know the weather | `{"intent":"WEATHER_QUERY","location_id":"location.home","period":"today"}` |
| UC02-WEATHER-B | 1 | UC-02 | what is the weather today | `{"intent":"WEATHER_QUERY","location_id":"location.home","period":"today"}` |
| UC02-WEATHER-B-N1 | 2 | UC-02 | can you tell me the weather today, please | `{"intent":"WEATHER_QUERY","location_id":"location.home","period":"today"}` |
| UC02-WEATHER-B-N2 | 2 | UC-02 | I would like to know the weather today | `{"intent":"WEATHER_QUERY","location_id":"location.home","period":"today"}` |
| UC03-ON-K-A | 1 | UC-03 | turn on kitchen lamp | `{"entity_id":"light.kitchen","intent":"POWER_ON"}` |
| UC03-ON-K-A-N1 | 2 | UC-03 | could you turn on kitchen lamp, please | `{"entity_id":"light.kitchen","intent":"POWER_ON"}` |
| UC03-ON-K-A-N2 | 2 | UC-03 | I would like you to turn on kitchen lamp for me | `{"entity_id":"light.kitchen","intent":"POWER_ON"}` |
| UC03-ON-K-B | 1 | UC-03 | turn kitchen lamp on | `{"entity_id":"light.kitchen","intent":"POWER_ON"}` |
| UC03-ON-K-B-N1 | 2 | UC-03 | could you turn kitchen lamp on, please | `{"entity_id":"light.kitchen","intent":"POWER_ON"}` |
| UC03-ON-K-B-N2 | 2 | UC-03 | I would like you to turn kitchen lamp on for me | `{"entity_id":"light.kitchen","intent":"POWER_ON"}` |
| UC03-OFF-K-A | 1 | UC-03 | turn off kitchen lamp | `{"entity_id":"light.kitchen","intent":"POWER_OFF"}` |
| UC03-OFF-K-A-N1 | 2 | UC-03 | could you turn off kitchen lamp, please | `{"entity_id":"light.kitchen","intent":"POWER_OFF"}` |
| UC03-OFF-K-A-N2 | 2 | UC-03 | I would like you to turn off kitchen lamp for me | `{"entity_id":"light.kitchen","intent":"POWER_OFF"}` |
| UC03-OFF-K-B | 1 | UC-03 | turn kitchen lamp off | `{"entity_id":"light.kitchen","intent":"POWER_OFF"}` |
| UC03-OFF-K-B-N1 | 2 | UC-03 | could you turn kitchen lamp off, please | `{"entity_id":"light.kitchen","intent":"POWER_OFF"}` |
| UC03-OFF-K-B-N2 | 2 | UC-03 | I would like you to turn kitchen lamp off for me | `{"entity_id":"light.kitchen","intent":"POWER_OFF"}` |
| UC04-BRI-K-000-A | 1 | UC-04 | set kitchen lamp to zero percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":0}` |
| UC04-BRI-K-000-A-N1 | 2 | UC-04 | could you set kitchen lamp to zero percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":0}` |
| UC04-BRI-K-000-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to zero percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":0}` |
| UC04-BRI-K-001-A | 1 | UC-04 | set kitchen lamp to one percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":1}` |
| UC04-BRI-K-001-A-N1 | 2 | UC-04 | could you set kitchen lamp to one percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":1}` |
| UC04-BRI-K-001-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to one percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":1}` |
| UC04-BRI-K-002-A | 1 | UC-04 | set kitchen lamp to two percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":2}` |
| UC04-BRI-K-002-A-N1 | 2 | UC-04 | could you set kitchen lamp to two percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":2}` |
| UC04-BRI-K-002-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to two percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":2}` |
| UC04-BRI-K-003-A | 1 | UC-04 | set kitchen lamp to three percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":3}` |
| UC04-BRI-K-003-A-N1 | 2 | UC-04 | could you set kitchen lamp to three percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":3}` |
| UC04-BRI-K-003-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to three percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":3}` |
| UC04-BRI-K-004-A | 1 | UC-04 | set kitchen lamp to four percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":4}` |
| UC04-BRI-K-004-A-N1 | 2 | UC-04 | could you set kitchen lamp to four percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":4}` |
| UC04-BRI-K-004-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to four percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":4}` |
| UC04-BRI-K-005-A | 1 | UC-04 | set kitchen lamp to five percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":5}` |
| UC04-BRI-K-005-A-N1 | 2 | UC-04 | could you set kitchen lamp to five percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":5}` |
| UC04-BRI-K-005-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to five percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":5}` |
| UC04-BRI-K-006-A | 1 | UC-04 | set kitchen lamp to six percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":6}` |
| UC04-BRI-K-006-A-N1 | 2 | UC-04 | could you set kitchen lamp to six percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":6}` |
| UC04-BRI-K-006-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to six percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":6}` |
| UC04-BRI-K-007-A | 1 | UC-04 | set kitchen lamp to seven percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":7}` |
| UC04-BRI-K-007-A-N1 | 2 | UC-04 | could you set kitchen lamp to seven percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":7}` |
| UC04-BRI-K-007-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to seven percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":7}` |
| UC04-BRI-K-008-A | 1 | UC-04 | set kitchen lamp to eight percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":8}` |
| UC04-BRI-K-008-A-N1 | 2 | UC-04 | could you set kitchen lamp to eight percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":8}` |
| UC04-BRI-K-008-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to eight percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":8}` |
| UC04-BRI-K-009-A | 1 | UC-04 | set kitchen lamp to nine percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":9}` |
| UC04-BRI-K-009-A-N1 | 2 | UC-04 | could you set kitchen lamp to nine percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":9}` |
| UC04-BRI-K-009-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to nine percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":9}` |
| UC04-BRI-K-010-A | 1 | UC-04 | set kitchen lamp to ten percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":10}` |
| UC04-BRI-K-010-A-N1 | 2 | UC-04 | could you set kitchen lamp to ten percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":10}` |
| UC04-BRI-K-010-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to ten percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":10}` |
| UC04-BRI-K-011-A | 1 | UC-04 | set kitchen lamp to eleven percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":11}` |
| UC04-BRI-K-011-A-N1 | 2 | UC-04 | could you set kitchen lamp to eleven percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":11}` |
| UC04-BRI-K-011-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to eleven percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":11}` |
| UC04-BRI-K-012-A | 1 | UC-04 | set kitchen lamp to twelve percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":12}` |
| UC04-BRI-K-012-A-N1 | 2 | UC-04 | could you set kitchen lamp to twelve percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":12}` |
| UC04-BRI-K-012-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to twelve percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":12}` |
| UC04-BRI-K-013-A | 1 | UC-04 | set kitchen lamp to thirteen percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":13}` |
| UC04-BRI-K-013-A-N1 | 2 | UC-04 | could you set kitchen lamp to thirteen percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":13}` |
| UC04-BRI-K-013-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to thirteen percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":13}` |
| UC04-BRI-K-014-A | 1 | UC-04 | set kitchen lamp to fourteen percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":14}` |
| UC04-BRI-K-014-A-N1 | 2 | UC-04 | could you set kitchen lamp to fourteen percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":14}` |
| UC04-BRI-K-014-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to fourteen percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":14}` |
| UC04-BRI-K-015-A | 1 | UC-04 | set kitchen lamp to fifteen percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":15}` |
| UC04-BRI-K-015-A-N1 | 2 | UC-04 | could you set kitchen lamp to fifteen percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":15}` |
| UC04-BRI-K-015-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to fifteen percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":15}` |
| UC04-BRI-K-016-A | 1 | UC-04 | set kitchen lamp to sixteen percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":16}` |
| UC04-BRI-K-016-A-N1 | 2 | UC-04 | could you set kitchen lamp to sixteen percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":16}` |
| UC04-BRI-K-016-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to sixteen percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":16}` |
| UC04-BRI-K-017-A | 1 | UC-04 | set kitchen lamp to seventeen percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":17}` |
| UC04-BRI-K-017-A-N1 | 2 | UC-04 | could you set kitchen lamp to seventeen percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":17}` |
| UC04-BRI-K-017-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to seventeen percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":17}` |
| UC04-BRI-K-018-A | 1 | UC-04 | set kitchen lamp to eighteen percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":18}` |
| UC04-BRI-K-018-A-N1 | 2 | UC-04 | could you set kitchen lamp to eighteen percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":18}` |
| UC04-BRI-K-018-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to eighteen percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":18}` |
| UC04-BRI-K-019-A | 1 | UC-04 | set kitchen lamp to nineteen percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":19}` |
| UC04-BRI-K-019-A-N1 | 2 | UC-04 | could you set kitchen lamp to nineteen percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":19}` |
| UC04-BRI-K-019-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to nineteen percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":19}` |
| UC04-BRI-K-020-A | 1 | UC-04 | set kitchen lamp to twenty percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":20}` |
| UC04-BRI-K-020-A-N1 | 2 | UC-04 | could you set kitchen lamp to twenty percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":20}` |
| UC04-BRI-K-020-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to twenty percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":20}` |
| UC04-BRI-K-021-A | 1 | UC-04 | set kitchen lamp to twenty one percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":21}` |
| UC04-BRI-K-021-A-N1 | 2 | UC-04 | could you set kitchen lamp to twenty one percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":21}` |
| UC04-BRI-K-021-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to twenty one percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":21}` |
| UC04-BRI-K-022-A | 1 | UC-04 | set kitchen lamp to twenty two percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":22}` |
| UC04-BRI-K-022-A-N1 | 2 | UC-04 | could you set kitchen lamp to twenty two percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":22}` |
| UC04-BRI-K-022-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to twenty two percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":22}` |
| UC04-BRI-K-023-A | 1 | UC-04 | set kitchen lamp to twenty three percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":23}` |
| UC04-BRI-K-023-A-N1 | 2 | UC-04 | could you set kitchen lamp to twenty three percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":23}` |
| UC04-BRI-K-023-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to twenty three percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":23}` |
| UC04-BRI-K-024-A | 1 | UC-04 | set kitchen lamp to twenty four percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":24}` |
| UC04-BRI-K-024-A-N1 | 2 | UC-04 | could you set kitchen lamp to twenty four percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":24}` |
| UC04-BRI-K-024-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to twenty four percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":24}` |
| UC04-BRI-K-025-A | 1 | UC-04 | set kitchen lamp to twenty five percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":25}` |
| UC04-BRI-K-025-A-N1 | 2 | UC-04 | could you set kitchen lamp to twenty five percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":25}` |
| UC04-BRI-K-025-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to twenty five percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":25}` |
| UC04-BRI-K-026-A | 1 | UC-04 | set kitchen lamp to twenty six percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":26}` |
| UC04-BRI-K-026-A-N1 | 2 | UC-04 | could you set kitchen lamp to twenty six percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":26}` |
| UC04-BRI-K-026-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to twenty six percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":26}` |
| UC04-BRI-K-027-A | 1 | UC-04 | set kitchen lamp to twenty seven percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":27}` |
| UC04-BRI-K-027-A-N1 | 2 | UC-04 | could you set kitchen lamp to twenty seven percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":27}` |
| UC04-BRI-K-027-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to twenty seven percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":27}` |
| UC04-BRI-K-028-A | 1 | UC-04 | set kitchen lamp to twenty eight percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":28}` |
| UC04-BRI-K-028-A-N1 | 2 | UC-04 | could you set kitchen lamp to twenty eight percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":28}` |
| UC04-BRI-K-028-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to twenty eight percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":28}` |
| UC04-BRI-K-029-A | 1 | UC-04 | set kitchen lamp to twenty nine percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":29}` |
| UC04-BRI-K-029-A-N1 | 2 | UC-04 | could you set kitchen lamp to twenty nine percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":29}` |
| UC04-BRI-K-029-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to twenty nine percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":29}` |
| UC04-BRI-K-030-A | 1 | UC-04 | set kitchen lamp to thirty percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":30}` |
| UC04-BRI-K-030-A-N1 | 2 | UC-04 | could you set kitchen lamp to thirty percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":30}` |
| UC04-BRI-K-030-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to thirty percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":30}` |
| UC04-BRI-K-031-A | 1 | UC-04 | set kitchen lamp to thirty one percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":31}` |
| UC04-BRI-K-031-A-N1 | 2 | UC-04 | could you set kitchen lamp to thirty one percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":31}` |
| UC04-BRI-K-031-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to thirty one percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":31}` |
| UC04-BRI-K-032-A | 1 | UC-04 | set kitchen lamp to thirty two percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":32}` |
| UC04-BRI-K-032-A-N1 | 2 | UC-04 | could you set kitchen lamp to thirty two percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":32}` |
| UC04-BRI-K-032-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to thirty two percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":32}` |
| UC04-BRI-K-033-A | 1 | UC-04 | set kitchen lamp to thirty three percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":33}` |
| UC04-BRI-K-033-A-N1 | 2 | UC-04 | could you set kitchen lamp to thirty three percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":33}` |
| UC04-BRI-K-033-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to thirty three percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":33}` |
| UC04-BRI-K-034-A | 1 | UC-04 | set kitchen lamp to thirty four percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":34}` |
| UC04-BRI-K-034-A-N1 | 2 | UC-04 | could you set kitchen lamp to thirty four percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":34}` |
| UC04-BRI-K-034-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to thirty four percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":34}` |
| UC04-BRI-K-035-A | 1 | UC-04 | set kitchen lamp to thirty five percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":35}` |
| UC04-BRI-K-035-A-N1 | 2 | UC-04 | could you set kitchen lamp to thirty five percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":35}` |
| UC04-BRI-K-035-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to thirty five percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":35}` |
| UC04-BRI-K-036-A | 1 | UC-04 | set kitchen lamp to thirty six percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":36}` |
| UC04-BRI-K-036-A-N1 | 2 | UC-04 | could you set kitchen lamp to thirty six percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":36}` |
| UC04-BRI-K-036-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to thirty six percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":36}` |
| UC04-BRI-K-037-A | 1 | UC-04 | set kitchen lamp to thirty seven percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":37}` |
| UC04-BRI-K-037-A-N1 | 2 | UC-04 | could you set kitchen lamp to thirty seven percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":37}` |
| UC04-BRI-K-037-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to thirty seven percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":37}` |
| UC04-BRI-K-038-A | 1 | UC-04 | set kitchen lamp to thirty eight percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":38}` |
| UC04-BRI-K-038-A-N1 | 2 | UC-04 | could you set kitchen lamp to thirty eight percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":38}` |
| UC04-BRI-K-038-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to thirty eight percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":38}` |
| UC04-BRI-K-039-A | 1 | UC-04 | set kitchen lamp to thirty nine percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":39}` |
| UC04-BRI-K-039-A-N1 | 2 | UC-04 | could you set kitchen lamp to thirty nine percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":39}` |
| UC04-BRI-K-039-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to thirty nine percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":39}` |
| UC04-BRI-K-040-A | 1 | UC-04 | set kitchen lamp to forty percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":40}` |
| UC04-BRI-K-040-A-N1 | 2 | UC-04 | could you set kitchen lamp to forty percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":40}` |
| UC04-BRI-K-040-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to forty percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":40}` |
| UC04-BRI-K-041-A | 1 | UC-04 | set kitchen lamp to forty one percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":41}` |
| UC04-BRI-K-041-A-N1 | 2 | UC-04 | could you set kitchen lamp to forty one percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":41}` |
| UC04-BRI-K-041-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to forty one percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":41}` |
| UC04-BRI-K-042-A | 1 | UC-04 | set kitchen lamp to forty two percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":42}` |
| UC04-BRI-K-042-A-N1 | 2 | UC-04 | could you set kitchen lamp to forty two percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":42}` |
| UC04-BRI-K-042-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to forty two percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":42}` |
| UC04-BRI-K-043-A | 1 | UC-04 | set kitchen lamp to forty three percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":43}` |
| UC04-BRI-K-043-A-N1 | 2 | UC-04 | could you set kitchen lamp to forty three percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":43}` |
| UC04-BRI-K-043-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to forty three percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":43}` |
| UC04-BRI-K-044-A | 1 | UC-04 | set kitchen lamp to forty four percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":44}` |
| UC04-BRI-K-044-A-N1 | 2 | UC-04 | could you set kitchen lamp to forty four percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":44}` |
| UC04-BRI-K-044-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to forty four percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":44}` |
| UC04-BRI-K-045-A | 1 | UC-04 | set kitchen lamp to forty five percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":45}` |
| UC04-BRI-K-045-A-N1 | 2 | UC-04 | could you set kitchen lamp to forty five percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":45}` |
| UC04-BRI-K-045-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to forty five percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":45}` |
| UC04-BRI-K-046-A | 1 | UC-04 | set kitchen lamp to forty six percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":46}` |
| UC04-BRI-K-046-A-N1 | 2 | UC-04 | could you set kitchen lamp to forty six percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":46}` |
| UC04-BRI-K-046-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to forty six percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":46}` |
| UC04-BRI-K-047-A | 1 | UC-04 | set kitchen lamp to forty seven percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":47}` |
| UC04-BRI-K-047-A-N1 | 2 | UC-04 | could you set kitchen lamp to forty seven percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":47}` |
| UC04-BRI-K-047-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to forty seven percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":47}` |
| UC04-BRI-K-048-A | 1 | UC-04 | set kitchen lamp to forty eight percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":48}` |
| UC04-BRI-K-048-A-N1 | 2 | UC-04 | could you set kitchen lamp to forty eight percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":48}` |
| UC04-BRI-K-048-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to forty eight percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":48}` |
| UC04-BRI-K-049-A | 1 | UC-04 | set kitchen lamp to forty nine percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":49}` |
| UC04-BRI-K-049-A-N1 | 2 | UC-04 | could you set kitchen lamp to forty nine percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":49}` |
| UC04-BRI-K-049-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to forty nine percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":49}` |
| UC04-BRI-K-050-A | 1 | UC-04 | set kitchen lamp to fifty percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":50}` |
| UC04-BRI-K-050-A-N1 | 2 | UC-04 | could you set kitchen lamp to fifty percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":50}` |
| UC04-BRI-K-050-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to fifty percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":50}` |
| UC04-BRI-K-051-A | 1 | UC-04 | set kitchen lamp to fifty one percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":51}` |
| UC04-BRI-K-051-A-N1 | 2 | UC-04 | could you set kitchen lamp to fifty one percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":51}` |
| UC04-BRI-K-051-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to fifty one percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":51}` |
| UC04-BRI-K-052-A | 1 | UC-04 | set kitchen lamp to fifty two percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":52}` |
| UC04-BRI-K-052-A-N1 | 2 | UC-04 | could you set kitchen lamp to fifty two percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":52}` |
| UC04-BRI-K-052-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to fifty two percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":52}` |
| UC04-BRI-K-053-A | 1 | UC-04 | set kitchen lamp to fifty three percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":53}` |
| UC04-BRI-K-053-A-N1 | 2 | UC-04 | could you set kitchen lamp to fifty three percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":53}` |
| UC04-BRI-K-053-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to fifty three percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":53}` |
| UC04-BRI-K-054-A | 1 | UC-04 | set kitchen lamp to fifty four percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":54}` |
| UC04-BRI-K-054-A-N1 | 2 | UC-04 | could you set kitchen lamp to fifty four percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":54}` |
| UC04-BRI-K-054-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to fifty four percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":54}` |
| UC04-BRI-K-055-A | 1 | UC-04 | set kitchen lamp to fifty five percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":55}` |
| UC04-BRI-K-055-A-N1 | 2 | UC-04 | could you set kitchen lamp to fifty five percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":55}` |
| UC04-BRI-K-055-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to fifty five percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":55}` |
| UC04-BRI-K-056-A | 1 | UC-04 | set kitchen lamp to fifty six percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":56}` |
| UC04-BRI-K-056-A-N1 | 2 | UC-04 | could you set kitchen lamp to fifty six percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":56}` |
| UC04-BRI-K-056-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to fifty six percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":56}` |
| UC04-BRI-K-057-A | 1 | UC-04 | set kitchen lamp to fifty seven percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":57}` |
| UC04-BRI-K-057-A-N1 | 2 | UC-04 | could you set kitchen lamp to fifty seven percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":57}` |
| UC04-BRI-K-057-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to fifty seven percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":57}` |
| UC04-BRI-K-058-A | 1 | UC-04 | set kitchen lamp to fifty eight percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":58}` |
| UC04-BRI-K-058-A-N1 | 2 | UC-04 | could you set kitchen lamp to fifty eight percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":58}` |
| UC04-BRI-K-058-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to fifty eight percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":58}` |
| UC04-BRI-K-059-A | 1 | UC-04 | set kitchen lamp to fifty nine percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":59}` |
| UC04-BRI-K-059-A-N1 | 2 | UC-04 | could you set kitchen lamp to fifty nine percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":59}` |
| UC04-BRI-K-059-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to fifty nine percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":59}` |
| UC04-BRI-K-060-A | 1 | UC-04 | set kitchen lamp to sixty percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":60}` |
| UC04-BRI-K-060-A-N1 | 2 | UC-04 | could you set kitchen lamp to sixty percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":60}` |
| UC04-BRI-K-060-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to sixty percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":60}` |
| UC04-BRI-K-061-A | 1 | UC-04 | set kitchen lamp to sixty one percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":61}` |
| UC04-BRI-K-061-A-N1 | 2 | UC-04 | could you set kitchen lamp to sixty one percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":61}` |
| UC04-BRI-K-061-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to sixty one percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":61}` |
| UC04-BRI-K-062-A | 1 | UC-04 | set kitchen lamp to sixty two percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":62}` |
| UC04-BRI-K-062-A-N1 | 2 | UC-04 | could you set kitchen lamp to sixty two percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":62}` |
| UC04-BRI-K-062-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to sixty two percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":62}` |
| UC04-BRI-K-063-A | 1 | UC-04 | set kitchen lamp to sixty three percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":63}` |
| UC04-BRI-K-063-A-N1 | 2 | UC-04 | could you set kitchen lamp to sixty three percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":63}` |
| UC04-BRI-K-063-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to sixty three percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":63}` |
| UC04-BRI-K-064-A | 1 | UC-04 | set kitchen lamp to sixty four percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":64}` |
| UC04-BRI-K-064-A-N1 | 2 | UC-04 | could you set kitchen lamp to sixty four percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":64}` |
| UC04-BRI-K-064-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to sixty four percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":64}` |
| UC04-BRI-K-065-A | 1 | UC-04 | set kitchen lamp to sixty five percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":65}` |
| UC04-BRI-K-065-A-N1 | 2 | UC-04 | could you set kitchen lamp to sixty five percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":65}` |
| UC04-BRI-K-065-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to sixty five percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":65}` |
| UC04-BRI-K-066-A | 1 | UC-04 | set kitchen lamp to sixty six percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":66}` |
| UC04-BRI-K-066-A-N1 | 2 | UC-04 | could you set kitchen lamp to sixty six percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":66}` |
| UC04-BRI-K-066-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to sixty six percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":66}` |
| UC04-BRI-K-067-A | 1 | UC-04 | set kitchen lamp to sixty seven percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":67}` |
| UC04-BRI-K-067-A-N1 | 2 | UC-04 | could you set kitchen lamp to sixty seven percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":67}` |
| UC04-BRI-K-067-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to sixty seven percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":67}` |
| UC04-BRI-K-068-A | 1 | UC-04 | set kitchen lamp to sixty eight percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":68}` |
| UC04-BRI-K-068-A-N1 | 2 | UC-04 | could you set kitchen lamp to sixty eight percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":68}` |
| UC04-BRI-K-068-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to sixty eight percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":68}` |
| UC04-BRI-K-069-A | 1 | UC-04 | set kitchen lamp to sixty nine percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":69}` |
| UC04-BRI-K-069-A-N1 | 2 | UC-04 | could you set kitchen lamp to sixty nine percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":69}` |
| UC04-BRI-K-069-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to sixty nine percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":69}` |
| UC04-BRI-K-070-A | 1 | UC-04 | set kitchen lamp to seventy percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":70}` |
| UC04-BRI-K-070-A-N1 | 2 | UC-04 | could you set kitchen lamp to seventy percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":70}` |
| UC04-BRI-K-070-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to seventy percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":70}` |
| UC04-BRI-K-071-A | 1 | UC-04 | set kitchen lamp to seventy one percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":71}` |
| UC04-BRI-K-071-A-N1 | 2 | UC-04 | could you set kitchen lamp to seventy one percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":71}` |
| UC04-BRI-K-071-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to seventy one percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":71}` |
| UC04-BRI-K-072-A | 1 | UC-04 | set kitchen lamp to seventy two percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":72}` |
| UC04-BRI-K-072-A-N1 | 2 | UC-04 | could you set kitchen lamp to seventy two percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":72}` |
| UC04-BRI-K-072-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to seventy two percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":72}` |
| UC04-BRI-K-073-A | 1 | UC-04 | set kitchen lamp to seventy three percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":73}` |
| UC04-BRI-K-073-A-N1 | 2 | UC-04 | could you set kitchen lamp to seventy three percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":73}` |
| UC04-BRI-K-073-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to seventy three percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":73}` |
| UC04-BRI-K-074-A | 1 | UC-04 | set kitchen lamp to seventy four percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":74}` |
| UC04-BRI-K-074-A-N1 | 2 | UC-04 | could you set kitchen lamp to seventy four percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":74}` |
| UC04-BRI-K-074-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to seventy four percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":74}` |
| UC04-BRI-K-075-A | 1 | UC-04 | set kitchen lamp to seventy five percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":75}` |
| UC04-BRI-K-075-A-N1 | 2 | UC-04 | could you set kitchen lamp to seventy five percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":75}` |
| UC04-BRI-K-075-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to seventy five percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":75}` |
| UC04-BRI-K-076-A | 1 | UC-04 | set kitchen lamp to seventy six percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":76}` |
| UC04-BRI-K-076-A-N1 | 2 | UC-04 | could you set kitchen lamp to seventy six percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":76}` |
| UC04-BRI-K-076-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to seventy six percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":76}` |
| UC04-BRI-K-077-A | 1 | UC-04 | set kitchen lamp to seventy seven percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":77}` |
| UC04-BRI-K-077-A-N1 | 2 | UC-04 | could you set kitchen lamp to seventy seven percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":77}` |
| UC04-BRI-K-077-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to seventy seven percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":77}` |
| UC04-BRI-K-078-A | 1 | UC-04 | set kitchen lamp to seventy eight percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":78}` |
| UC04-BRI-K-078-A-N1 | 2 | UC-04 | could you set kitchen lamp to seventy eight percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":78}` |
| UC04-BRI-K-078-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to seventy eight percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":78}` |
| UC04-BRI-K-079-A | 1 | UC-04 | set kitchen lamp to seventy nine percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":79}` |
| UC04-BRI-K-079-A-N1 | 2 | UC-04 | could you set kitchen lamp to seventy nine percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":79}` |
| UC04-BRI-K-079-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to seventy nine percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":79}` |
| UC04-BRI-K-080-A | 1 | UC-04 | set kitchen lamp to eighty percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":80}` |
| UC04-BRI-K-080-A-N1 | 2 | UC-04 | could you set kitchen lamp to eighty percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":80}` |
| UC04-BRI-K-080-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to eighty percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":80}` |
| UC04-BRI-K-081-A | 1 | UC-04 | set kitchen lamp to eighty one percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":81}` |
| UC04-BRI-K-081-A-N1 | 2 | UC-04 | could you set kitchen lamp to eighty one percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":81}` |
| UC04-BRI-K-081-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to eighty one percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":81}` |
| UC04-BRI-K-082-A | 1 | UC-04 | set kitchen lamp to eighty two percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":82}` |
| UC04-BRI-K-082-A-N1 | 2 | UC-04 | could you set kitchen lamp to eighty two percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":82}` |
| UC04-BRI-K-082-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to eighty two percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":82}` |
| UC04-BRI-K-083-A | 1 | UC-04 | set kitchen lamp to eighty three percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":83}` |
| UC04-BRI-K-083-A-N1 | 2 | UC-04 | could you set kitchen lamp to eighty three percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":83}` |
| UC04-BRI-K-083-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to eighty three percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":83}` |
| UC04-BRI-K-084-A | 1 | UC-04 | set kitchen lamp to eighty four percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":84}` |
| UC04-BRI-K-084-A-N1 | 2 | UC-04 | could you set kitchen lamp to eighty four percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":84}` |
| UC04-BRI-K-084-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to eighty four percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":84}` |
| UC04-BRI-K-085-A | 1 | UC-04 | set kitchen lamp to eighty five percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":85}` |
| UC04-BRI-K-085-A-N1 | 2 | UC-04 | could you set kitchen lamp to eighty five percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":85}` |
| UC04-BRI-K-085-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to eighty five percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":85}` |
| UC04-BRI-K-086-A | 1 | UC-04 | set kitchen lamp to eighty six percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":86}` |
| UC04-BRI-K-086-A-N1 | 2 | UC-04 | could you set kitchen lamp to eighty six percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":86}` |
| UC04-BRI-K-086-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to eighty six percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":86}` |
| UC04-BRI-K-087-A | 1 | UC-04 | set kitchen lamp to eighty seven percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":87}` |
| UC04-BRI-K-087-A-N1 | 2 | UC-04 | could you set kitchen lamp to eighty seven percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":87}` |
| UC04-BRI-K-087-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to eighty seven percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":87}` |
| UC04-BRI-K-088-A | 1 | UC-04 | set kitchen lamp to eighty eight percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":88}` |
| UC04-BRI-K-088-A-N1 | 2 | UC-04 | could you set kitchen lamp to eighty eight percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":88}` |
| UC04-BRI-K-088-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to eighty eight percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":88}` |
| UC04-BRI-K-089-A | 1 | UC-04 | set kitchen lamp to eighty nine percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":89}` |
| UC04-BRI-K-089-A-N1 | 2 | UC-04 | could you set kitchen lamp to eighty nine percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":89}` |
| UC04-BRI-K-089-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to eighty nine percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":89}` |
| UC04-BRI-K-090-A | 1 | UC-04 | set kitchen lamp to ninety percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":90}` |
| UC04-BRI-K-090-A-N1 | 2 | UC-04 | could you set kitchen lamp to ninety percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":90}` |
| UC04-BRI-K-090-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to ninety percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":90}` |
| UC04-BRI-K-091-A | 1 | UC-04 | set kitchen lamp to ninety one percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":91}` |
| UC04-BRI-K-091-A-N1 | 2 | UC-04 | could you set kitchen lamp to ninety one percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":91}` |
| UC04-BRI-K-091-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to ninety one percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":91}` |
| UC04-BRI-K-092-A | 1 | UC-04 | set kitchen lamp to ninety two percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":92}` |
| UC04-BRI-K-092-A-N1 | 2 | UC-04 | could you set kitchen lamp to ninety two percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":92}` |
| UC04-BRI-K-092-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to ninety two percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":92}` |
| UC04-BRI-K-093-A | 1 | UC-04 | set kitchen lamp to ninety three percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":93}` |
| UC04-BRI-K-093-A-N1 | 2 | UC-04 | could you set kitchen lamp to ninety three percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":93}` |
| UC04-BRI-K-093-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to ninety three percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":93}` |
| UC04-BRI-K-094-A | 1 | UC-04 | set kitchen lamp to ninety four percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":94}` |
| UC04-BRI-K-094-A-N1 | 2 | UC-04 | could you set kitchen lamp to ninety four percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":94}` |
| UC04-BRI-K-094-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to ninety four percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":94}` |
| UC04-BRI-K-095-A | 1 | UC-04 | set kitchen lamp to ninety five percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":95}` |
| UC04-BRI-K-095-A-N1 | 2 | UC-04 | could you set kitchen lamp to ninety five percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":95}` |
| UC04-BRI-K-095-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to ninety five percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":95}` |
| UC04-BRI-K-096-A | 1 | UC-04 | set kitchen lamp to ninety six percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":96}` |
| UC04-BRI-K-096-A-N1 | 2 | UC-04 | could you set kitchen lamp to ninety six percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":96}` |
| UC04-BRI-K-096-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to ninety six percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":96}` |
| UC04-BRI-K-097-A | 1 | UC-04 | set kitchen lamp to ninety seven percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":97}` |
| UC04-BRI-K-097-A-N1 | 2 | UC-04 | could you set kitchen lamp to ninety seven percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":97}` |
| UC04-BRI-K-097-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to ninety seven percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":97}` |
| UC04-BRI-K-098-A | 1 | UC-04 | set kitchen lamp to ninety eight percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":98}` |
| UC04-BRI-K-098-A-N1 | 2 | UC-04 | could you set kitchen lamp to ninety eight percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":98}` |
| UC04-BRI-K-098-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to ninety eight percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":98}` |
| UC04-BRI-K-099-A | 1 | UC-04 | set kitchen lamp to ninety nine percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":99}` |
| UC04-BRI-K-099-A-N1 | 2 | UC-04 | could you set kitchen lamp to ninety nine percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":99}` |
| UC04-BRI-K-099-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to ninety nine percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":99}` |
| UC04-BRI-K-100-A | 1 | UC-04 | set kitchen lamp to one hundred percent | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":100}` |
| UC04-BRI-K-100-A-N1 | 2 | UC-04 | could you set kitchen lamp to one hundred percent, please | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":100}` |
| UC04-BRI-K-100-A-N2 | 2 | UC-04 | I would like you to set kitchen lamp to one hundred percent for me | `{"entity_id":"light.kitchen","intent":"BRIGHTNESS","value":100}` |
| UC04-DELTA-K-P05 | 1 | UC-04 | increase kitchen lamp by five percent | `{"delta":5,"entity_id":"light.kitchen","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-K-P05-N1 | 2 | UC-04 | could you increase kitchen lamp by five percent, please | `{"delta":5,"entity_id":"light.kitchen","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-K-P05-N2 | 2 | UC-04 | I would like you to increase kitchen lamp by five percent for me | `{"delta":5,"entity_id":"light.kitchen","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-K-P10 | 1 | UC-04 | increase kitchen lamp by ten percent | `{"delta":10,"entity_id":"light.kitchen","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-K-P10-N1 | 2 | UC-04 | could you increase kitchen lamp by ten percent, please | `{"delta":10,"entity_id":"light.kitchen","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-K-P10-N2 | 2 | UC-04 | I would like you to increase kitchen lamp by ten percent for me | `{"delta":10,"entity_id":"light.kitchen","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-K-M05 | 1 | UC-04 | decrease kitchen lamp by five percent | `{"delta":-5,"entity_id":"light.kitchen","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-K-M05-N1 | 2 | UC-04 | could you decrease kitchen lamp by five percent, please | `{"delta":-5,"entity_id":"light.kitchen","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-K-M05-N2 | 2 | UC-04 | I would like you to decrease kitchen lamp by five percent for me | `{"delta":-5,"entity_id":"light.kitchen","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-K-M10 | 1 | UC-04 | decrease kitchen lamp by ten percent | `{"delta":-10,"entity_id":"light.kitchen","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-K-M10-N1 | 2 | UC-04 | could you decrease kitchen lamp by ten percent, please | `{"delta":-10,"entity_id":"light.kitchen","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-K-M10-N2 | 2 | UC-04 | I would like you to decrease kitchen lamp by ten percent for me | `{"delta":-10,"entity_id":"light.kitchen","intent":"BRIGHTNESS_DELTA"}` |
| UC03-ON-B-A | 1 | UC-03 | turn on bedroom lamp | `{"entity_id":"light.bedroom","intent":"POWER_ON"}` |
| UC03-ON-B-A-N1 | 2 | UC-03 | could you turn on bedroom lamp, please | `{"entity_id":"light.bedroom","intent":"POWER_ON"}` |
| UC03-ON-B-A-N2 | 2 | UC-03 | I would like you to turn on bedroom lamp for me | `{"entity_id":"light.bedroom","intent":"POWER_ON"}` |
| UC03-ON-B-B | 1 | UC-03 | turn bedroom lamp on | `{"entity_id":"light.bedroom","intent":"POWER_ON"}` |
| UC03-ON-B-B-N1 | 2 | UC-03 | could you turn bedroom lamp on, please | `{"entity_id":"light.bedroom","intent":"POWER_ON"}` |
| UC03-ON-B-B-N2 | 2 | UC-03 | I would like you to turn bedroom lamp on for me | `{"entity_id":"light.bedroom","intent":"POWER_ON"}` |
| UC03-OFF-B-A | 1 | UC-03 | turn off bedroom lamp | `{"entity_id":"light.bedroom","intent":"POWER_OFF"}` |
| UC03-OFF-B-A-N1 | 2 | UC-03 | could you turn off bedroom lamp, please | `{"entity_id":"light.bedroom","intent":"POWER_OFF"}` |
| UC03-OFF-B-A-N2 | 2 | UC-03 | I would like you to turn off bedroom lamp for me | `{"entity_id":"light.bedroom","intent":"POWER_OFF"}` |
| UC03-OFF-B-B | 1 | UC-03 | turn bedroom lamp off | `{"entity_id":"light.bedroom","intent":"POWER_OFF"}` |
| UC03-OFF-B-B-N1 | 2 | UC-03 | could you turn bedroom lamp off, please | `{"entity_id":"light.bedroom","intent":"POWER_OFF"}` |
| UC03-OFF-B-B-N2 | 2 | UC-03 | I would like you to turn bedroom lamp off for me | `{"entity_id":"light.bedroom","intent":"POWER_OFF"}` |
| UC04-BRI-B-000-A | 1 | UC-04 | set bedroom lamp to zero percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":0}` |
| UC04-BRI-B-000-A-N1 | 2 | UC-04 | could you set bedroom lamp to zero percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":0}` |
| UC04-BRI-B-000-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to zero percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":0}` |
| UC04-BRI-B-001-A | 1 | UC-04 | set bedroom lamp to one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":1}` |
| UC04-BRI-B-001-A-N1 | 2 | UC-04 | could you set bedroom lamp to one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":1}` |
| UC04-BRI-B-001-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":1}` |
| UC04-BRI-B-002-A | 1 | UC-04 | set bedroom lamp to two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":2}` |
| UC04-BRI-B-002-A-N1 | 2 | UC-04 | could you set bedroom lamp to two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":2}` |
| UC04-BRI-B-002-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":2}` |
| UC04-BRI-B-003-A | 1 | UC-04 | set bedroom lamp to three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":3}` |
| UC04-BRI-B-003-A-N1 | 2 | UC-04 | could you set bedroom lamp to three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":3}` |
| UC04-BRI-B-003-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":3}` |
| UC04-BRI-B-004-A | 1 | UC-04 | set bedroom lamp to four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":4}` |
| UC04-BRI-B-004-A-N1 | 2 | UC-04 | could you set bedroom lamp to four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":4}` |
| UC04-BRI-B-004-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":4}` |
| UC04-BRI-B-005-A | 1 | UC-04 | set bedroom lamp to five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":5}` |
| UC04-BRI-B-005-A-N1 | 2 | UC-04 | could you set bedroom lamp to five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":5}` |
| UC04-BRI-B-005-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":5}` |
| UC04-BRI-B-006-A | 1 | UC-04 | set bedroom lamp to six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":6}` |
| UC04-BRI-B-006-A-N1 | 2 | UC-04 | could you set bedroom lamp to six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":6}` |
| UC04-BRI-B-006-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":6}` |
| UC04-BRI-B-007-A | 1 | UC-04 | set bedroom lamp to seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":7}` |
| UC04-BRI-B-007-A-N1 | 2 | UC-04 | could you set bedroom lamp to seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":7}` |
| UC04-BRI-B-007-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":7}` |
| UC04-BRI-B-008-A | 1 | UC-04 | set bedroom lamp to eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":8}` |
| UC04-BRI-B-008-A-N1 | 2 | UC-04 | could you set bedroom lamp to eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":8}` |
| UC04-BRI-B-008-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":8}` |
| UC04-BRI-B-009-A | 1 | UC-04 | set bedroom lamp to nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":9}` |
| UC04-BRI-B-009-A-N1 | 2 | UC-04 | could you set bedroom lamp to nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":9}` |
| UC04-BRI-B-009-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":9}` |
| UC04-BRI-B-010-A | 1 | UC-04 | set bedroom lamp to ten percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":10}` |
| UC04-BRI-B-010-A-N1 | 2 | UC-04 | could you set bedroom lamp to ten percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":10}` |
| UC04-BRI-B-010-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to ten percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":10}` |
| UC04-BRI-B-011-A | 1 | UC-04 | set bedroom lamp to eleven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":11}` |
| UC04-BRI-B-011-A-N1 | 2 | UC-04 | could you set bedroom lamp to eleven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":11}` |
| UC04-BRI-B-011-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to eleven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":11}` |
| UC04-BRI-B-012-A | 1 | UC-04 | set bedroom lamp to twelve percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":12}` |
| UC04-BRI-B-012-A-N1 | 2 | UC-04 | could you set bedroom lamp to twelve percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":12}` |
| UC04-BRI-B-012-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to twelve percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":12}` |
| UC04-BRI-B-013-A | 1 | UC-04 | set bedroom lamp to thirteen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":13}` |
| UC04-BRI-B-013-A-N1 | 2 | UC-04 | could you set bedroom lamp to thirteen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":13}` |
| UC04-BRI-B-013-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to thirteen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":13}` |
| UC04-BRI-B-014-A | 1 | UC-04 | set bedroom lamp to fourteen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":14}` |
| UC04-BRI-B-014-A-N1 | 2 | UC-04 | could you set bedroom lamp to fourteen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":14}` |
| UC04-BRI-B-014-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to fourteen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":14}` |
| UC04-BRI-B-015-A | 1 | UC-04 | set bedroom lamp to fifteen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":15}` |
| UC04-BRI-B-015-A-N1 | 2 | UC-04 | could you set bedroom lamp to fifteen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":15}` |
| UC04-BRI-B-015-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to fifteen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":15}` |
| UC04-BRI-B-016-A | 1 | UC-04 | set bedroom lamp to sixteen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":16}` |
| UC04-BRI-B-016-A-N1 | 2 | UC-04 | could you set bedroom lamp to sixteen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":16}` |
| UC04-BRI-B-016-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to sixteen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":16}` |
| UC04-BRI-B-017-A | 1 | UC-04 | set bedroom lamp to seventeen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":17}` |
| UC04-BRI-B-017-A-N1 | 2 | UC-04 | could you set bedroom lamp to seventeen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":17}` |
| UC04-BRI-B-017-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to seventeen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":17}` |
| UC04-BRI-B-018-A | 1 | UC-04 | set bedroom lamp to eighteen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":18}` |
| UC04-BRI-B-018-A-N1 | 2 | UC-04 | could you set bedroom lamp to eighteen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":18}` |
| UC04-BRI-B-018-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to eighteen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":18}` |
| UC04-BRI-B-019-A | 1 | UC-04 | set bedroom lamp to nineteen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":19}` |
| UC04-BRI-B-019-A-N1 | 2 | UC-04 | could you set bedroom lamp to nineteen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":19}` |
| UC04-BRI-B-019-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to nineteen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":19}` |
| UC04-BRI-B-020-A | 1 | UC-04 | set bedroom lamp to twenty percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":20}` |
| UC04-BRI-B-020-A-N1 | 2 | UC-04 | could you set bedroom lamp to twenty percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":20}` |
| UC04-BRI-B-020-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to twenty percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":20}` |
| UC04-BRI-B-021-A | 1 | UC-04 | set bedroom lamp to twenty one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":21}` |
| UC04-BRI-B-021-A-N1 | 2 | UC-04 | could you set bedroom lamp to twenty one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":21}` |
| UC04-BRI-B-021-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to twenty one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":21}` |
| UC04-BRI-B-022-A | 1 | UC-04 | set bedroom lamp to twenty two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":22}` |
| UC04-BRI-B-022-A-N1 | 2 | UC-04 | could you set bedroom lamp to twenty two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":22}` |
| UC04-BRI-B-022-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to twenty two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":22}` |
| UC04-BRI-B-023-A | 1 | UC-04 | set bedroom lamp to twenty three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":23}` |
| UC04-BRI-B-023-A-N1 | 2 | UC-04 | could you set bedroom lamp to twenty three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":23}` |
| UC04-BRI-B-023-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to twenty three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":23}` |
| UC04-BRI-B-024-A | 1 | UC-04 | set bedroom lamp to twenty four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":24}` |
| UC04-BRI-B-024-A-N1 | 2 | UC-04 | could you set bedroom lamp to twenty four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":24}` |
| UC04-BRI-B-024-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to twenty four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":24}` |
| UC04-BRI-B-025-A | 1 | UC-04 | set bedroom lamp to twenty five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":25}` |
| UC04-BRI-B-025-A-N1 | 2 | UC-04 | could you set bedroom lamp to twenty five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":25}` |
| UC04-BRI-B-025-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to twenty five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":25}` |
| UC04-BRI-B-026-A | 1 | UC-04 | set bedroom lamp to twenty six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":26}` |
| UC04-BRI-B-026-A-N1 | 2 | UC-04 | could you set bedroom lamp to twenty six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":26}` |
| UC04-BRI-B-026-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to twenty six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":26}` |
| UC04-BRI-B-027-A | 1 | UC-04 | set bedroom lamp to twenty seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":27}` |
| UC04-BRI-B-027-A-N1 | 2 | UC-04 | could you set bedroom lamp to twenty seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":27}` |
| UC04-BRI-B-027-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to twenty seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":27}` |
| UC04-BRI-B-028-A | 1 | UC-04 | set bedroom lamp to twenty eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":28}` |
| UC04-BRI-B-028-A-N1 | 2 | UC-04 | could you set bedroom lamp to twenty eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":28}` |
| UC04-BRI-B-028-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to twenty eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":28}` |
| UC04-BRI-B-029-A | 1 | UC-04 | set bedroom lamp to twenty nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":29}` |
| UC04-BRI-B-029-A-N1 | 2 | UC-04 | could you set bedroom lamp to twenty nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":29}` |
| UC04-BRI-B-029-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to twenty nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":29}` |
| UC04-BRI-B-030-A | 1 | UC-04 | set bedroom lamp to thirty percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":30}` |
| UC04-BRI-B-030-A-N1 | 2 | UC-04 | could you set bedroom lamp to thirty percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":30}` |
| UC04-BRI-B-030-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to thirty percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":30}` |
| UC04-BRI-B-031-A | 1 | UC-04 | set bedroom lamp to thirty one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":31}` |
| UC04-BRI-B-031-A-N1 | 2 | UC-04 | could you set bedroom lamp to thirty one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":31}` |
| UC04-BRI-B-031-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to thirty one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":31}` |
| UC04-BRI-B-032-A | 1 | UC-04 | set bedroom lamp to thirty two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":32}` |
| UC04-BRI-B-032-A-N1 | 2 | UC-04 | could you set bedroom lamp to thirty two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":32}` |
| UC04-BRI-B-032-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to thirty two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":32}` |
| UC04-BRI-B-033-A | 1 | UC-04 | set bedroom lamp to thirty three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":33}` |
| UC04-BRI-B-033-A-N1 | 2 | UC-04 | could you set bedroom lamp to thirty three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":33}` |
| UC04-BRI-B-033-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to thirty three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":33}` |
| UC04-BRI-B-034-A | 1 | UC-04 | set bedroom lamp to thirty four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":34}` |
| UC04-BRI-B-034-A-N1 | 2 | UC-04 | could you set bedroom lamp to thirty four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":34}` |
| UC04-BRI-B-034-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to thirty four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":34}` |
| UC04-BRI-B-035-A | 1 | UC-04 | set bedroom lamp to thirty five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":35}` |
| UC04-BRI-B-035-A-N1 | 2 | UC-04 | could you set bedroom lamp to thirty five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":35}` |
| UC04-BRI-B-035-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to thirty five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":35}` |
| UC04-BRI-B-036-A | 1 | UC-04 | set bedroom lamp to thirty six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":36}` |
| UC04-BRI-B-036-A-N1 | 2 | UC-04 | could you set bedroom lamp to thirty six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":36}` |
| UC04-BRI-B-036-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to thirty six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":36}` |
| UC04-BRI-B-037-A | 1 | UC-04 | set bedroom lamp to thirty seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":37}` |
| UC04-BRI-B-037-A-N1 | 2 | UC-04 | could you set bedroom lamp to thirty seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":37}` |
| UC04-BRI-B-037-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to thirty seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":37}` |
| UC04-BRI-B-038-A | 1 | UC-04 | set bedroom lamp to thirty eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":38}` |
| UC04-BRI-B-038-A-N1 | 2 | UC-04 | could you set bedroom lamp to thirty eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":38}` |
| UC04-BRI-B-038-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to thirty eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":38}` |
| UC04-BRI-B-039-A | 1 | UC-04 | set bedroom lamp to thirty nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":39}` |
| UC04-BRI-B-039-A-N1 | 2 | UC-04 | could you set bedroom lamp to thirty nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":39}` |
| UC04-BRI-B-039-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to thirty nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":39}` |
| UC04-BRI-B-040-A | 1 | UC-04 | set bedroom lamp to forty percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":40}` |
| UC04-BRI-B-040-A-N1 | 2 | UC-04 | could you set bedroom lamp to forty percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":40}` |
| UC04-BRI-B-040-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to forty percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":40}` |
| UC04-BRI-B-041-A | 1 | UC-04 | set bedroom lamp to forty one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":41}` |
| UC04-BRI-B-041-A-N1 | 2 | UC-04 | could you set bedroom lamp to forty one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":41}` |
| UC04-BRI-B-041-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to forty one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":41}` |
| UC04-BRI-B-042-A | 1 | UC-04 | set bedroom lamp to forty two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":42}` |
| UC04-BRI-B-042-A-N1 | 2 | UC-04 | could you set bedroom lamp to forty two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":42}` |
| UC04-BRI-B-042-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to forty two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":42}` |
| UC04-BRI-B-043-A | 1 | UC-04 | set bedroom lamp to forty three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":43}` |
| UC04-BRI-B-043-A-N1 | 2 | UC-04 | could you set bedroom lamp to forty three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":43}` |
| UC04-BRI-B-043-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to forty three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":43}` |
| UC04-BRI-B-044-A | 1 | UC-04 | set bedroom lamp to forty four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":44}` |
| UC04-BRI-B-044-A-N1 | 2 | UC-04 | could you set bedroom lamp to forty four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":44}` |
| UC04-BRI-B-044-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to forty four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":44}` |
| UC04-BRI-B-045-A | 1 | UC-04 | set bedroom lamp to forty five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":45}` |
| UC04-BRI-B-045-A-N1 | 2 | UC-04 | could you set bedroom lamp to forty five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":45}` |
| UC04-BRI-B-045-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to forty five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":45}` |
| UC04-BRI-B-046-A | 1 | UC-04 | set bedroom lamp to forty six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":46}` |
| UC04-BRI-B-046-A-N1 | 2 | UC-04 | could you set bedroom lamp to forty six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":46}` |
| UC04-BRI-B-046-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to forty six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":46}` |
| UC04-BRI-B-047-A | 1 | UC-04 | set bedroom lamp to forty seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":47}` |
| UC04-BRI-B-047-A-N1 | 2 | UC-04 | could you set bedroom lamp to forty seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":47}` |
| UC04-BRI-B-047-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to forty seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":47}` |
| UC04-BRI-B-048-A | 1 | UC-04 | set bedroom lamp to forty eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":48}` |
| UC04-BRI-B-048-A-N1 | 2 | UC-04 | could you set bedroom lamp to forty eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":48}` |
| UC04-BRI-B-048-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to forty eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":48}` |
| UC04-BRI-B-049-A | 1 | UC-04 | set bedroom lamp to forty nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":49}` |
| UC04-BRI-B-049-A-N1 | 2 | UC-04 | could you set bedroom lamp to forty nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":49}` |
| UC04-BRI-B-049-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to forty nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":49}` |
| UC04-BRI-B-050-A | 1 | UC-04 | set bedroom lamp to fifty percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":50}` |
| UC04-BRI-B-050-A-N1 | 2 | UC-04 | could you set bedroom lamp to fifty percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":50}` |
| UC04-BRI-B-050-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to fifty percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":50}` |
| UC04-BRI-B-051-A | 1 | UC-04 | set bedroom lamp to fifty one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":51}` |
| UC04-BRI-B-051-A-N1 | 2 | UC-04 | could you set bedroom lamp to fifty one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":51}` |
| UC04-BRI-B-051-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to fifty one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":51}` |
| UC04-BRI-B-052-A | 1 | UC-04 | set bedroom lamp to fifty two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":52}` |
| UC04-BRI-B-052-A-N1 | 2 | UC-04 | could you set bedroom lamp to fifty two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":52}` |
| UC04-BRI-B-052-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to fifty two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":52}` |
| UC04-BRI-B-053-A | 1 | UC-04 | set bedroom lamp to fifty three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":53}` |
| UC04-BRI-B-053-A-N1 | 2 | UC-04 | could you set bedroom lamp to fifty three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":53}` |
| UC04-BRI-B-053-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to fifty three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":53}` |
| UC04-BRI-B-054-A | 1 | UC-04 | set bedroom lamp to fifty four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":54}` |
| UC04-BRI-B-054-A-N1 | 2 | UC-04 | could you set bedroom lamp to fifty four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":54}` |
| UC04-BRI-B-054-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to fifty four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":54}` |
| UC04-BRI-B-055-A | 1 | UC-04 | set bedroom lamp to fifty five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":55}` |
| UC04-BRI-B-055-A-N1 | 2 | UC-04 | could you set bedroom lamp to fifty five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":55}` |
| UC04-BRI-B-055-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to fifty five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":55}` |
| UC04-BRI-B-056-A | 1 | UC-04 | set bedroom lamp to fifty six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":56}` |
| UC04-BRI-B-056-A-N1 | 2 | UC-04 | could you set bedroom lamp to fifty six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":56}` |
| UC04-BRI-B-056-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to fifty six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":56}` |
| UC04-BRI-B-057-A | 1 | UC-04 | set bedroom lamp to fifty seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":57}` |
| UC04-BRI-B-057-A-N1 | 2 | UC-04 | could you set bedroom lamp to fifty seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":57}` |
| UC04-BRI-B-057-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to fifty seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":57}` |
| UC04-BRI-B-058-A | 1 | UC-04 | set bedroom lamp to fifty eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":58}` |
| UC04-BRI-B-058-A-N1 | 2 | UC-04 | could you set bedroom lamp to fifty eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":58}` |
| UC04-BRI-B-058-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to fifty eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":58}` |
| UC04-BRI-B-059-A | 1 | UC-04 | set bedroom lamp to fifty nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":59}` |
| UC04-BRI-B-059-A-N1 | 2 | UC-04 | could you set bedroom lamp to fifty nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":59}` |
| UC04-BRI-B-059-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to fifty nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":59}` |
| UC04-BRI-B-060-A | 1 | UC-04 | set bedroom lamp to sixty percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":60}` |
| UC04-BRI-B-060-A-N1 | 2 | UC-04 | could you set bedroom lamp to sixty percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":60}` |
| UC04-BRI-B-060-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to sixty percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":60}` |
| UC04-BRI-B-061-A | 1 | UC-04 | set bedroom lamp to sixty one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":61}` |
| UC04-BRI-B-061-A-N1 | 2 | UC-04 | could you set bedroom lamp to sixty one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":61}` |
| UC04-BRI-B-061-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to sixty one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":61}` |
| UC04-BRI-B-062-A | 1 | UC-04 | set bedroom lamp to sixty two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":62}` |
| UC04-BRI-B-062-A-N1 | 2 | UC-04 | could you set bedroom lamp to sixty two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":62}` |
| UC04-BRI-B-062-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to sixty two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":62}` |
| UC04-BRI-B-063-A | 1 | UC-04 | set bedroom lamp to sixty three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":63}` |
| UC04-BRI-B-063-A-N1 | 2 | UC-04 | could you set bedroom lamp to sixty three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":63}` |
| UC04-BRI-B-063-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to sixty three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":63}` |
| UC04-BRI-B-064-A | 1 | UC-04 | set bedroom lamp to sixty four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":64}` |
| UC04-BRI-B-064-A-N1 | 2 | UC-04 | could you set bedroom lamp to sixty four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":64}` |
| UC04-BRI-B-064-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to sixty four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":64}` |
| UC04-BRI-B-065-A | 1 | UC-04 | set bedroom lamp to sixty five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":65}` |
| UC04-BRI-B-065-A-N1 | 2 | UC-04 | could you set bedroom lamp to sixty five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":65}` |
| UC04-BRI-B-065-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to sixty five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":65}` |
| UC04-BRI-B-066-A | 1 | UC-04 | set bedroom lamp to sixty six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":66}` |
| UC04-BRI-B-066-A-N1 | 2 | UC-04 | could you set bedroom lamp to sixty six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":66}` |
| UC04-BRI-B-066-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to sixty six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":66}` |
| UC04-BRI-B-067-A | 1 | UC-04 | set bedroom lamp to sixty seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":67}` |
| UC04-BRI-B-067-A-N1 | 2 | UC-04 | could you set bedroom lamp to sixty seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":67}` |
| UC04-BRI-B-067-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to sixty seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":67}` |
| UC04-BRI-B-068-A | 1 | UC-04 | set bedroom lamp to sixty eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":68}` |
| UC04-BRI-B-068-A-N1 | 2 | UC-04 | could you set bedroom lamp to sixty eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":68}` |
| UC04-BRI-B-068-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to sixty eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":68}` |
| UC04-BRI-B-069-A | 1 | UC-04 | set bedroom lamp to sixty nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":69}` |
| UC04-BRI-B-069-A-N1 | 2 | UC-04 | could you set bedroom lamp to sixty nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":69}` |
| UC04-BRI-B-069-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to sixty nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":69}` |
| UC04-BRI-B-070-A | 1 | UC-04 | set bedroom lamp to seventy percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":70}` |
| UC04-BRI-B-070-A-N1 | 2 | UC-04 | could you set bedroom lamp to seventy percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":70}` |
| UC04-BRI-B-070-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to seventy percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":70}` |
| UC04-BRI-B-071-A | 1 | UC-04 | set bedroom lamp to seventy one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":71}` |
| UC04-BRI-B-071-A-N1 | 2 | UC-04 | could you set bedroom lamp to seventy one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":71}` |
| UC04-BRI-B-071-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to seventy one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":71}` |
| UC04-BRI-B-072-A | 1 | UC-04 | set bedroom lamp to seventy two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":72}` |
| UC04-BRI-B-072-A-N1 | 2 | UC-04 | could you set bedroom lamp to seventy two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":72}` |
| UC04-BRI-B-072-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to seventy two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":72}` |
| UC04-BRI-B-073-A | 1 | UC-04 | set bedroom lamp to seventy three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":73}` |
| UC04-BRI-B-073-A-N1 | 2 | UC-04 | could you set bedroom lamp to seventy three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":73}` |
| UC04-BRI-B-073-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to seventy three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":73}` |
| UC04-BRI-B-074-A | 1 | UC-04 | set bedroom lamp to seventy four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":74}` |
| UC04-BRI-B-074-A-N1 | 2 | UC-04 | could you set bedroom lamp to seventy four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":74}` |
| UC04-BRI-B-074-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to seventy four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":74}` |
| UC04-BRI-B-075-A | 1 | UC-04 | set bedroom lamp to seventy five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":75}` |
| UC04-BRI-B-075-A-N1 | 2 | UC-04 | could you set bedroom lamp to seventy five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":75}` |
| UC04-BRI-B-075-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to seventy five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":75}` |
| UC04-BRI-B-076-A | 1 | UC-04 | set bedroom lamp to seventy six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":76}` |
| UC04-BRI-B-076-A-N1 | 2 | UC-04 | could you set bedroom lamp to seventy six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":76}` |
| UC04-BRI-B-076-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to seventy six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":76}` |
| UC04-BRI-B-077-A | 1 | UC-04 | set bedroom lamp to seventy seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":77}` |
| UC04-BRI-B-077-A-N1 | 2 | UC-04 | could you set bedroom lamp to seventy seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":77}` |
| UC04-BRI-B-077-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to seventy seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":77}` |
| UC04-BRI-B-078-A | 1 | UC-04 | set bedroom lamp to seventy eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":78}` |
| UC04-BRI-B-078-A-N1 | 2 | UC-04 | could you set bedroom lamp to seventy eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":78}` |
| UC04-BRI-B-078-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to seventy eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":78}` |
| UC04-BRI-B-079-A | 1 | UC-04 | set bedroom lamp to seventy nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":79}` |
| UC04-BRI-B-079-A-N1 | 2 | UC-04 | could you set bedroom lamp to seventy nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":79}` |
| UC04-BRI-B-079-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to seventy nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":79}` |
| UC04-BRI-B-080-A | 1 | UC-04 | set bedroom lamp to eighty percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":80}` |
| UC04-BRI-B-080-A-N1 | 2 | UC-04 | could you set bedroom lamp to eighty percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":80}` |
| UC04-BRI-B-080-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to eighty percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":80}` |
| UC04-BRI-B-081-A | 1 | UC-04 | set bedroom lamp to eighty one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":81}` |
| UC04-BRI-B-081-A-N1 | 2 | UC-04 | could you set bedroom lamp to eighty one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":81}` |
| UC04-BRI-B-081-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to eighty one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":81}` |
| UC04-BRI-B-082-A | 1 | UC-04 | set bedroom lamp to eighty two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":82}` |
| UC04-BRI-B-082-A-N1 | 2 | UC-04 | could you set bedroom lamp to eighty two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":82}` |
| UC04-BRI-B-082-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to eighty two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":82}` |
| UC04-BRI-B-083-A | 1 | UC-04 | set bedroom lamp to eighty three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":83}` |
| UC04-BRI-B-083-A-N1 | 2 | UC-04 | could you set bedroom lamp to eighty three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":83}` |
| UC04-BRI-B-083-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to eighty three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":83}` |
| UC04-BRI-B-084-A | 1 | UC-04 | set bedroom lamp to eighty four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":84}` |
| UC04-BRI-B-084-A-N1 | 2 | UC-04 | could you set bedroom lamp to eighty four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":84}` |
| UC04-BRI-B-084-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to eighty four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":84}` |
| UC04-BRI-B-085-A | 1 | UC-04 | set bedroom lamp to eighty five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":85}` |
| UC04-BRI-B-085-A-N1 | 2 | UC-04 | could you set bedroom lamp to eighty five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":85}` |
| UC04-BRI-B-085-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to eighty five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":85}` |
| UC04-BRI-B-086-A | 1 | UC-04 | set bedroom lamp to eighty six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":86}` |
| UC04-BRI-B-086-A-N1 | 2 | UC-04 | could you set bedroom lamp to eighty six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":86}` |
| UC04-BRI-B-086-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to eighty six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":86}` |
| UC04-BRI-B-087-A | 1 | UC-04 | set bedroom lamp to eighty seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":87}` |
| UC04-BRI-B-087-A-N1 | 2 | UC-04 | could you set bedroom lamp to eighty seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":87}` |
| UC04-BRI-B-087-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to eighty seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":87}` |
| UC04-BRI-B-088-A | 1 | UC-04 | set bedroom lamp to eighty eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":88}` |
| UC04-BRI-B-088-A-N1 | 2 | UC-04 | could you set bedroom lamp to eighty eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":88}` |
| UC04-BRI-B-088-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to eighty eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":88}` |
| UC04-BRI-B-089-A | 1 | UC-04 | set bedroom lamp to eighty nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":89}` |
| UC04-BRI-B-089-A-N1 | 2 | UC-04 | could you set bedroom lamp to eighty nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":89}` |
| UC04-BRI-B-089-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to eighty nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":89}` |
| UC04-BRI-B-090-A | 1 | UC-04 | set bedroom lamp to ninety percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":90}` |
| UC04-BRI-B-090-A-N1 | 2 | UC-04 | could you set bedroom lamp to ninety percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":90}` |
| UC04-BRI-B-090-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to ninety percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":90}` |
| UC04-BRI-B-091-A | 1 | UC-04 | set bedroom lamp to ninety one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":91}` |
| UC04-BRI-B-091-A-N1 | 2 | UC-04 | could you set bedroom lamp to ninety one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":91}` |
| UC04-BRI-B-091-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to ninety one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":91}` |
| UC04-BRI-B-092-A | 1 | UC-04 | set bedroom lamp to ninety two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":92}` |
| UC04-BRI-B-092-A-N1 | 2 | UC-04 | could you set bedroom lamp to ninety two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":92}` |
| UC04-BRI-B-092-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to ninety two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":92}` |
| UC04-BRI-B-093-A | 1 | UC-04 | set bedroom lamp to ninety three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":93}` |
| UC04-BRI-B-093-A-N1 | 2 | UC-04 | could you set bedroom lamp to ninety three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":93}` |
| UC04-BRI-B-093-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to ninety three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":93}` |
| UC04-BRI-B-094-A | 1 | UC-04 | set bedroom lamp to ninety four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":94}` |
| UC04-BRI-B-094-A-N1 | 2 | UC-04 | could you set bedroom lamp to ninety four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":94}` |
| UC04-BRI-B-094-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to ninety four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":94}` |
| UC04-BRI-B-095-A | 1 | UC-04 | set bedroom lamp to ninety five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":95}` |
| UC04-BRI-B-095-A-N1 | 2 | UC-04 | could you set bedroom lamp to ninety five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":95}` |
| UC04-BRI-B-095-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to ninety five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":95}` |
| UC04-BRI-B-096-A | 1 | UC-04 | set bedroom lamp to ninety six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":96}` |
| UC04-BRI-B-096-A-N1 | 2 | UC-04 | could you set bedroom lamp to ninety six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":96}` |
| UC04-BRI-B-096-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to ninety six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":96}` |
| UC04-BRI-B-097-A | 1 | UC-04 | set bedroom lamp to ninety seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":97}` |
| UC04-BRI-B-097-A-N1 | 2 | UC-04 | could you set bedroom lamp to ninety seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":97}` |
| UC04-BRI-B-097-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to ninety seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":97}` |
| UC04-BRI-B-098-A | 1 | UC-04 | set bedroom lamp to ninety eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":98}` |
| UC04-BRI-B-098-A-N1 | 2 | UC-04 | could you set bedroom lamp to ninety eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":98}` |
| UC04-BRI-B-098-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to ninety eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":98}` |
| UC04-BRI-B-099-A | 1 | UC-04 | set bedroom lamp to ninety nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":99}` |
| UC04-BRI-B-099-A-N1 | 2 | UC-04 | could you set bedroom lamp to ninety nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":99}` |
| UC04-BRI-B-099-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to ninety nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":99}` |
| UC04-BRI-B-100-A | 1 | UC-04 | set bedroom lamp to one hundred percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":100}` |
| UC04-BRI-B-100-A-N1 | 2 | UC-04 | could you set bedroom lamp to one hundred percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":100}` |
| UC04-BRI-B-100-A-N2 | 2 | UC-04 | I would like you to set bedroom lamp to one hundred percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":100}` |
| UC04-DELTA-B-P05 | 1 | UC-04 | increase bedroom lamp by five percent | `{"delta":5,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B-P05-N1 | 2 | UC-04 | could you increase bedroom lamp by five percent, please | `{"delta":5,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B-P05-N2 | 2 | UC-04 | I would like you to increase bedroom lamp by five percent for me | `{"delta":5,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B-P10 | 1 | UC-04 | increase bedroom lamp by ten percent | `{"delta":10,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B-P10-N1 | 2 | UC-04 | could you increase bedroom lamp by ten percent, please | `{"delta":10,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B-P10-N2 | 2 | UC-04 | I would like you to increase bedroom lamp by ten percent for me | `{"delta":10,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B-M05 | 1 | UC-04 | decrease bedroom lamp by five percent | `{"delta":-5,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B-M05-N1 | 2 | UC-04 | could you decrease bedroom lamp by five percent, please | `{"delta":-5,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B-M05-N2 | 2 | UC-04 | I would like you to decrease bedroom lamp by five percent for me | `{"delta":-5,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B-M10 | 1 | UC-04 | decrease bedroom lamp by ten percent | `{"delta":-10,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B-M10-N1 | 2 | UC-04 | could you decrease bedroom lamp by ten percent, please | `{"delta":-10,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B-M10-N2 | 2 | UC-04 | I would like you to decrease bedroom lamp by ten percent for me | `{"delta":-10,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC03-ON-B2-A | 1 | UC-03 | turn on lamp two | `{"entity_id":"light.bedroom","intent":"POWER_ON"}` |
| UC03-ON-B2-A-N1 | 2 | UC-03 | could you turn on lamp two, please | `{"entity_id":"light.bedroom","intent":"POWER_ON"}` |
| UC03-ON-B2-A-N2 | 2 | UC-03 | I would like you to turn on lamp two for me | `{"entity_id":"light.bedroom","intent":"POWER_ON"}` |
| UC03-ON-B2-B | 1 | UC-03 | turn lamp two on | `{"entity_id":"light.bedroom","intent":"POWER_ON"}` |
| UC03-ON-B2-B-N1 | 2 | UC-03 | could you turn lamp two on, please | `{"entity_id":"light.bedroom","intent":"POWER_ON"}` |
| UC03-ON-B2-B-N2 | 2 | UC-03 | I would like you to turn lamp two on for me | `{"entity_id":"light.bedroom","intent":"POWER_ON"}` |
| UC03-OFF-B2-A | 1 | UC-03 | turn off lamp two | `{"entity_id":"light.bedroom","intent":"POWER_OFF"}` |
| UC03-OFF-B2-A-N1 | 2 | UC-03 | could you turn off lamp two, please | `{"entity_id":"light.bedroom","intent":"POWER_OFF"}` |
| UC03-OFF-B2-A-N2 | 2 | UC-03 | I would like you to turn off lamp two for me | `{"entity_id":"light.bedroom","intent":"POWER_OFF"}` |
| UC03-OFF-B2-B | 1 | UC-03 | turn lamp two off | `{"entity_id":"light.bedroom","intent":"POWER_OFF"}` |
| UC03-OFF-B2-B-N1 | 2 | UC-03 | could you turn lamp two off, please | `{"entity_id":"light.bedroom","intent":"POWER_OFF"}` |
| UC03-OFF-B2-B-N2 | 2 | UC-03 | I would like you to turn lamp two off for me | `{"entity_id":"light.bedroom","intent":"POWER_OFF"}` |
| UC04-BRI-B2-000-A | 1 | UC-04 | set lamp two to zero percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":0}` |
| UC04-BRI-B2-000-A-N1 | 2 | UC-04 | could you set lamp two to zero percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":0}` |
| UC04-BRI-B2-000-A-N2 | 2 | UC-04 | I would like you to set lamp two to zero percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":0}` |
| UC04-BRI-B2-001-A | 1 | UC-04 | set lamp two to one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":1}` |
| UC04-BRI-B2-001-A-N1 | 2 | UC-04 | could you set lamp two to one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":1}` |
| UC04-BRI-B2-001-A-N2 | 2 | UC-04 | I would like you to set lamp two to one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":1}` |
| UC04-BRI-B2-002-A | 1 | UC-04 | set lamp two to two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":2}` |
| UC04-BRI-B2-002-A-N1 | 2 | UC-04 | could you set lamp two to two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":2}` |
| UC04-BRI-B2-002-A-N2 | 2 | UC-04 | I would like you to set lamp two to two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":2}` |
| UC04-BRI-B2-003-A | 1 | UC-04 | set lamp two to three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":3}` |
| UC04-BRI-B2-003-A-N1 | 2 | UC-04 | could you set lamp two to three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":3}` |
| UC04-BRI-B2-003-A-N2 | 2 | UC-04 | I would like you to set lamp two to three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":3}` |
| UC04-BRI-B2-004-A | 1 | UC-04 | set lamp two to four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":4}` |
| UC04-BRI-B2-004-A-N1 | 2 | UC-04 | could you set lamp two to four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":4}` |
| UC04-BRI-B2-004-A-N2 | 2 | UC-04 | I would like you to set lamp two to four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":4}` |
| UC04-BRI-B2-005-A | 1 | UC-04 | set lamp two to five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":5}` |
| UC04-BRI-B2-005-A-N1 | 2 | UC-04 | could you set lamp two to five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":5}` |
| UC04-BRI-B2-005-A-N2 | 2 | UC-04 | I would like you to set lamp two to five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":5}` |
| UC04-BRI-B2-006-A | 1 | UC-04 | set lamp two to six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":6}` |
| UC04-BRI-B2-006-A-N1 | 2 | UC-04 | could you set lamp two to six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":6}` |
| UC04-BRI-B2-006-A-N2 | 2 | UC-04 | I would like you to set lamp two to six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":6}` |
| UC04-BRI-B2-007-A | 1 | UC-04 | set lamp two to seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":7}` |
| UC04-BRI-B2-007-A-N1 | 2 | UC-04 | could you set lamp two to seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":7}` |
| UC04-BRI-B2-007-A-N2 | 2 | UC-04 | I would like you to set lamp two to seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":7}` |
| UC04-BRI-B2-008-A | 1 | UC-04 | set lamp two to eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":8}` |
| UC04-BRI-B2-008-A-N1 | 2 | UC-04 | could you set lamp two to eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":8}` |
| UC04-BRI-B2-008-A-N2 | 2 | UC-04 | I would like you to set lamp two to eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":8}` |
| UC04-BRI-B2-009-A | 1 | UC-04 | set lamp two to nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":9}` |
| UC04-BRI-B2-009-A-N1 | 2 | UC-04 | could you set lamp two to nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":9}` |
| UC04-BRI-B2-009-A-N2 | 2 | UC-04 | I would like you to set lamp two to nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":9}` |
| UC04-BRI-B2-010-A | 1 | UC-04 | set lamp two to ten percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":10}` |
| UC04-BRI-B2-010-A-N1 | 2 | UC-04 | could you set lamp two to ten percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":10}` |
| UC04-BRI-B2-010-A-N2 | 2 | UC-04 | I would like you to set lamp two to ten percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":10}` |
| UC04-BRI-B2-011-A | 1 | UC-04 | set lamp two to eleven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":11}` |
| UC04-BRI-B2-011-A-N1 | 2 | UC-04 | could you set lamp two to eleven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":11}` |
| UC04-BRI-B2-011-A-N2 | 2 | UC-04 | I would like you to set lamp two to eleven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":11}` |
| UC04-BRI-B2-012-A | 1 | UC-04 | set lamp two to twelve percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":12}` |
| UC04-BRI-B2-012-A-N1 | 2 | UC-04 | could you set lamp two to twelve percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":12}` |
| UC04-BRI-B2-012-A-N2 | 2 | UC-04 | I would like you to set lamp two to twelve percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":12}` |
| UC04-BRI-B2-013-A | 1 | UC-04 | set lamp two to thirteen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":13}` |
| UC04-BRI-B2-013-A-N1 | 2 | UC-04 | could you set lamp two to thirteen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":13}` |
| UC04-BRI-B2-013-A-N2 | 2 | UC-04 | I would like you to set lamp two to thirteen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":13}` |
| UC04-BRI-B2-014-A | 1 | UC-04 | set lamp two to fourteen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":14}` |
| UC04-BRI-B2-014-A-N1 | 2 | UC-04 | could you set lamp two to fourteen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":14}` |
| UC04-BRI-B2-014-A-N2 | 2 | UC-04 | I would like you to set lamp two to fourteen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":14}` |
| UC04-BRI-B2-015-A | 1 | UC-04 | set lamp two to fifteen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":15}` |
| UC04-BRI-B2-015-A-N1 | 2 | UC-04 | could you set lamp two to fifteen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":15}` |
| UC04-BRI-B2-015-A-N2 | 2 | UC-04 | I would like you to set lamp two to fifteen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":15}` |
| UC04-BRI-B2-016-A | 1 | UC-04 | set lamp two to sixteen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":16}` |
| UC04-BRI-B2-016-A-N1 | 2 | UC-04 | could you set lamp two to sixteen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":16}` |
| UC04-BRI-B2-016-A-N2 | 2 | UC-04 | I would like you to set lamp two to sixteen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":16}` |
| UC04-BRI-B2-017-A | 1 | UC-04 | set lamp two to seventeen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":17}` |
| UC04-BRI-B2-017-A-N1 | 2 | UC-04 | could you set lamp two to seventeen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":17}` |
| UC04-BRI-B2-017-A-N2 | 2 | UC-04 | I would like you to set lamp two to seventeen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":17}` |
| UC04-BRI-B2-018-A | 1 | UC-04 | set lamp two to eighteen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":18}` |
| UC04-BRI-B2-018-A-N1 | 2 | UC-04 | could you set lamp two to eighteen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":18}` |
| UC04-BRI-B2-018-A-N2 | 2 | UC-04 | I would like you to set lamp two to eighteen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":18}` |
| UC04-BRI-B2-019-A | 1 | UC-04 | set lamp two to nineteen percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":19}` |
| UC04-BRI-B2-019-A-N1 | 2 | UC-04 | could you set lamp two to nineteen percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":19}` |
| UC04-BRI-B2-019-A-N2 | 2 | UC-04 | I would like you to set lamp two to nineteen percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":19}` |
| UC04-BRI-B2-020-A | 1 | UC-04 | set lamp two to twenty percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":20}` |
| UC04-BRI-B2-020-A-N1 | 2 | UC-04 | could you set lamp two to twenty percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":20}` |
| UC04-BRI-B2-020-A-N2 | 2 | UC-04 | I would like you to set lamp two to twenty percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":20}` |
| UC04-BRI-B2-021-A | 1 | UC-04 | set lamp two to twenty one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":21}` |
| UC04-BRI-B2-021-A-N1 | 2 | UC-04 | could you set lamp two to twenty one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":21}` |
| UC04-BRI-B2-021-A-N2 | 2 | UC-04 | I would like you to set lamp two to twenty one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":21}` |
| UC04-BRI-B2-022-A | 1 | UC-04 | set lamp two to twenty two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":22}` |
| UC04-BRI-B2-022-A-N1 | 2 | UC-04 | could you set lamp two to twenty two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":22}` |
| UC04-BRI-B2-022-A-N2 | 2 | UC-04 | I would like you to set lamp two to twenty two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":22}` |
| UC04-BRI-B2-023-A | 1 | UC-04 | set lamp two to twenty three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":23}` |
| UC04-BRI-B2-023-A-N1 | 2 | UC-04 | could you set lamp two to twenty three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":23}` |
| UC04-BRI-B2-023-A-N2 | 2 | UC-04 | I would like you to set lamp two to twenty three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":23}` |
| UC04-BRI-B2-024-A | 1 | UC-04 | set lamp two to twenty four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":24}` |
| UC04-BRI-B2-024-A-N1 | 2 | UC-04 | could you set lamp two to twenty four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":24}` |
| UC04-BRI-B2-024-A-N2 | 2 | UC-04 | I would like you to set lamp two to twenty four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":24}` |
| UC04-BRI-B2-025-A | 1 | UC-04 | set lamp two to twenty five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":25}` |
| UC04-BRI-B2-025-A-N1 | 2 | UC-04 | could you set lamp two to twenty five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":25}` |
| UC04-BRI-B2-025-A-N2 | 2 | UC-04 | I would like you to set lamp two to twenty five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":25}` |
| UC04-BRI-B2-026-A | 1 | UC-04 | set lamp two to twenty six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":26}` |
| UC04-BRI-B2-026-A-N1 | 2 | UC-04 | could you set lamp two to twenty six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":26}` |
| UC04-BRI-B2-026-A-N2 | 2 | UC-04 | I would like you to set lamp two to twenty six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":26}` |
| UC04-BRI-B2-027-A | 1 | UC-04 | set lamp two to twenty seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":27}` |
| UC04-BRI-B2-027-A-N1 | 2 | UC-04 | could you set lamp two to twenty seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":27}` |
| UC04-BRI-B2-027-A-N2 | 2 | UC-04 | I would like you to set lamp two to twenty seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":27}` |
| UC04-BRI-B2-028-A | 1 | UC-04 | set lamp two to twenty eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":28}` |
| UC04-BRI-B2-028-A-N1 | 2 | UC-04 | could you set lamp two to twenty eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":28}` |
| UC04-BRI-B2-028-A-N2 | 2 | UC-04 | I would like you to set lamp two to twenty eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":28}` |
| UC04-BRI-B2-029-A | 1 | UC-04 | set lamp two to twenty nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":29}` |
| UC04-BRI-B2-029-A-N1 | 2 | UC-04 | could you set lamp two to twenty nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":29}` |
| UC04-BRI-B2-029-A-N2 | 2 | UC-04 | I would like you to set lamp two to twenty nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":29}` |
| UC04-BRI-B2-030-A | 1 | UC-04 | set lamp two to thirty percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":30}` |
| UC04-BRI-B2-030-A-N1 | 2 | UC-04 | could you set lamp two to thirty percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":30}` |
| UC04-BRI-B2-030-A-N2 | 2 | UC-04 | I would like you to set lamp two to thirty percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":30}` |
| UC04-BRI-B2-031-A | 1 | UC-04 | set lamp two to thirty one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":31}` |
| UC04-BRI-B2-031-A-N1 | 2 | UC-04 | could you set lamp two to thirty one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":31}` |
| UC04-BRI-B2-031-A-N2 | 2 | UC-04 | I would like you to set lamp two to thirty one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":31}` |
| UC04-BRI-B2-032-A | 1 | UC-04 | set lamp two to thirty two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":32}` |
| UC04-BRI-B2-032-A-N1 | 2 | UC-04 | could you set lamp two to thirty two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":32}` |
| UC04-BRI-B2-032-A-N2 | 2 | UC-04 | I would like you to set lamp two to thirty two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":32}` |
| UC04-BRI-B2-033-A | 1 | UC-04 | set lamp two to thirty three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":33}` |
| UC04-BRI-B2-033-A-N1 | 2 | UC-04 | could you set lamp two to thirty three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":33}` |
| UC04-BRI-B2-033-A-N2 | 2 | UC-04 | I would like you to set lamp two to thirty three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":33}` |
| UC04-BRI-B2-034-A | 1 | UC-04 | set lamp two to thirty four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":34}` |
| UC04-BRI-B2-034-A-N1 | 2 | UC-04 | could you set lamp two to thirty four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":34}` |
| UC04-BRI-B2-034-A-N2 | 2 | UC-04 | I would like you to set lamp two to thirty four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":34}` |
| UC04-BRI-B2-035-A | 1 | UC-04 | set lamp two to thirty five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":35}` |
| UC04-BRI-B2-035-A-N1 | 2 | UC-04 | could you set lamp two to thirty five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":35}` |
| UC04-BRI-B2-035-A-N2 | 2 | UC-04 | I would like you to set lamp two to thirty five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":35}` |
| UC04-BRI-B2-036-A | 1 | UC-04 | set lamp two to thirty six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":36}` |
| UC04-BRI-B2-036-A-N1 | 2 | UC-04 | could you set lamp two to thirty six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":36}` |
| UC04-BRI-B2-036-A-N2 | 2 | UC-04 | I would like you to set lamp two to thirty six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":36}` |
| UC04-BRI-B2-037-A | 1 | UC-04 | set lamp two to thirty seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":37}` |
| UC04-BRI-B2-037-A-N1 | 2 | UC-04 | could you set lamp two to thirty seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":37}` |
| UC04-BRI-B2-037-A-N2 | 2 | UC-04 | I would like you to set lamp two to thirty seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":37}` |
| UC04-BRI-B2-038-A | 1 | UC-04 | set lamp two to thirty eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":38}` |
| UC04-BRI-B2-038-A-N1 | 2 | UC-04 | could you set lamp two to thirty eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":38}` |
| UC04-BRI-B2-038-A-N2 | 2 | UC-04 | I would like you to set lamp two to thirty eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":38}` |
| UC04-BRI-B2-039-A | 1 | UC-04 | set lamp two to thirty nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":39}` |
| UC04-BRI-B2-039-A-N1 | 2 | UC-04 | could you set lamp two to thirty nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":39}` |
| UC04-BRI-B2-039-A-N2 | 2 | UC-04 | I would like you to set lamp two to thirty nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":39}` |
| UC04-BRI-B2-040-A | 1 | UC-04 | set lamp two to forty percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":40}` |
| UC04-BRI-B2-040-A-N1 | 2 | UC-04 | could you set lamp two to forty percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":40}` |
| UC04-BRI-B2-040-A-N2 | 2 | UC-04 | I would like you to set lamp two to forty percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":40}` |
| UC04-BRI-B2-041-A | 1 | UC-04 | set lamp two to forty one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":41}` |
| UC04-BRI-B2-041-A-N1 | 2 | UC-04 | could you set lamp two to forty one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":41}` |
| UC04-BRI-B2-041-A-N2 | 2 | UC-04 | I would like you to set lamp two to forty one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":41}` |
| UC04-BRI-B2-042-A | 1 | UC-04 | set lamp two to forty two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":42}` |
| UC04-BRI-B2-042-A-N1 | 2 | UC-04 | could you set lamp two to forty two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":42}` |
| UC04-BRI-B2-042-A-N2 | 2 | UC-04 | I would like you to set lamp two to forty two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":42}` |
| UC04-BRI-B2-043-A | 1 | UC-04 | set lamp two to forty three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":43}` |
| UC04-BRI-B2-043-A-N1 | 2 | UC-04 | could you set lamp two to forty three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":43}` |
| UC04-BRI-B2-043-A-N2 | 2 | UC-04 | I would like you to set lamp two to forty three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":43}` |
| UC04-BRI-B2-044-A | 1 | UC-04 | set lamp two to forty four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":44}` |
| UC04-BRI-B2-044-A-N1 | 2 | UC-04 | could you set lamp two to forty four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":44}` |
| UC04-BRI-B2-044-A-N2 | 2 | UC-04 | I would like you to set lamp two to forty four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":44}` |
| UC04-BRI-B2-045-A | 1 | UC-04 | set lamp two to forty five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":45}` |
| UC04-BRI-B2-045-A-N1 | 2 | UC-04 | could you set lamp two to forty five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":45}` |
| UC04-BRI-B2-045-A-N2 | 2 | UC-04 | I would like you to set lamp two to forty five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":45}` |
| UC04-BRI-B2-046-A | 1 | UC-04 | set lamp two to forty six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":46}` |
| UC04-BRI-B2-046-A-N1 | 2 | UC-04 | could you set lamp two to forty six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":46}` |
| UC04-BRI-B2-046-A-N2 | 2 | UC-04 | I would like you to set lamp two to forty six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":46}` |
| UC04-BRI-B2-047-A | 1 | UC-04 | set lamp two to forty seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":47}` |
| UC04-BRI-B2-047-A-N1 | 2 | UC-04 | could you set lamp two to forty seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":47}` |
| UC04-BRI-B2-047-A-N2 | 2 | UC-04 | I would like you to set lamp two to forty seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":47}` |
| UC04-BRI-B2-048-A | 1 | UC-04 | set lamp two to forty eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":48}` |
| UC04-BRI-B2-048-A-N1 | 2 | UC-04 | could you set lamp two to forty eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":48}` |
| UC04-BRI-B2-048-A-N2 | 2 | UC-04 | I would like you to set lamp two to forty eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":48}` |
| UC04-BRI-B2-049-A | 1 | UC-04 | set lamp two to forty nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":49}` |
| UC04-BRI-B2-049-A-N1 | 2 | UC-04 | could you set lamp two to forty nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":49}` |
| UC04-BRI-B2-049-A-N2 | 2 | UC-04 | I would like you to set lamp two to forty nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":49}` |
| UC04-BRI-B2-050-A | 1 | UC-04 | set lamp two to fifty percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":50}` |
| UC04-BRI-B2-050-A-N1 | 2 | UC-04 | could you set lamp two to fifty percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":50}` |
| UC04-BRI-B2-050-A-N2 | 2 | UC-04 | I would like you to set lamp two to fifty percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":50}` |
| UC04-BRI-B2-051-A | 1 | UC-04 | set lamp two to fifty one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":51}` |
| UC04-BRI-B2-051-A-N1 | 2 | UC-04 | could you set lamp two to fifty one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":51}` |
| UC04-BRI-B2-051-A-N2 | 2 | UC-04 | I would like you to set lamp two to fifty one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":51}` |
| UC04-BRI-B2-052-A | 1 | UC-04 | set lamp two to fifty two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":52}` |
| UC04-BRI-B2-052-A-N1 | 2 | UC-04 | could you set lamp two to fifty two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":52}` |
| UC04-BRI-B2-052-A-N2 | 2 | UC-04 | I would like you to set lamp two to fifty two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":52}` |
| UC04-BRI-B2-053-A | 1 | UC-04 | set lamp two to fifty three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":53}` |
| UC04-BRI-B2-053-A-N1 | 2 | UC-04 | could you set lamp two to fifty three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":53}` |
| UC04-BRI-B2-053-A-N2 | 2 | UC-04 | I would like you to set lamp two to fifty three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":53}` |
| UC04-BRI-B2-054-A | 1 | UC-04 | set lamp two to fifty four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":54}` |
| UC04-BRI-B2-054-A-N1 | 2 | UC-04 | could you set lamp two to fifty four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":54}` |
| UC04-BRI-B2-054-A-N2 | 2 | UC-04 | I would like you to set lamp two to fifty four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":54}` |
| UC04-BRI-B2-055-A | 1 | UC-04 | set lamp two to fifty five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":55}` |
| UC04-BRI-B2-055-A-N1 | 2 | UC-04 | could you set lamp two to fifty five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":55}` |
| UC04-BRI-B2-055-A-N2 | 2 | UC-04 | I would like you to set lamp two to fifty five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":55}` |
| UC04-BRI-B2-056-A | 1 | UC-04 | set lamp two to fifty six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":56}` |
| UC04-BRI-B2-056-A-N1 | 2 | UC-04 | could you set lamp two to fifty six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":56}` |
| UC04-BRI-B2-056-A-N2 | 2 | UC-04 | I would like you to set lamp two to fifty six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":56}` |
| UC04-BRI-B2-057-A | 1 | UC-04 | set lamp two to fifty seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":57}` |
| UC04-BRI-B2-057-A-N1 | 2 | UC-04 | could you set lamp two to fifty seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":57}` |
| UC04-BRI-B2-057-A-N2 | 2 | UC-04 | I would like you to set lamp two to fifty seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":57}` |
| UC04-BRI-B2-058-A | 1 | UC-04 | set lamp two to fifty eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":58}` |
| UC04-BRI-B2-058-A-N1 | 2 | UC-04 | could you set lamp two to fifty eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":58}` |
| UC04-BRI-B2-058-A-N2 | 2 | UC-04 | I would like you to set lamp two to fifty eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":58}` |
| UC04-BRI-B2-059-A | 1 | UC-04 | set lamp two to fifty nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":59}` |
| UC04-BRI-B2-059-A-N1 | 2 | UC-04 | could you set lamp two to fifty nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":59}` |
| UC04-BRI-B2-059-A-N2 | 2 | UC-04 | I would like you to set lamp two to fifty nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":59}` |
| UC04-BRI-B2-060-A | 1 | UC-04 | set lamp two to sixty percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":60}` |
| UC04-BRI-B2-060-A-N1 | 2 | UC-04 | could you set lamp two to sixty percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":60}` |
| UC04-BRI-B2-060-A-N2 | 2 | UC-04 | I would like you to set lamp two to sixty percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":60}` |
| UC04-BRI-B2-061-A | 1 | UC-04 | set lamp two to sixty one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":61}` |
| UC04-BRI-B2-061-A-N1 | 2 | UC-04 | could you set lamp two to sixty one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":61}` |
| UC04-BRI-B2-061-A-N2 | 2 | UC-04 | I would like you to set lamp two to sixty one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":61}` |
| UC04-BRI-B2-062-A | 1 | UC-04 | set lamp two to sixty two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":62}` |
| UC04-BRI-B2-062-A-N1 | 2 | UC-04 | could you set lamp two to sixty two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":62}` |
| UC04-BRI-B2-062-A-N2 | 2 | UC-04 | I would like you to set lamp two to sixty two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":62}` |
| UC04-BRI-B2-063-A | 1 | UC-04 | set lamp two to sixty three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":63}` |
| UC04-BRI-B2-063-A-N1 | 2 | UC-04 | could you set lamp two to sixty three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":63}` |
| UC04-BRI-B2-063-A-N2 | 2 | UC-04 | I would like you to set lamp two to sixty three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":63}` |
| UC04-BRI-B2-064-A | 1 | UC-04 | set lamp two to sixty four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":64}` |
| UC04-BRI-B2-064-A-N1 | 2 | UC-04 | could you set lamp two to sixty four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":64}` |
| UC04-BRI-B2-064-A-N2 | 2 | UC-04 | I would like you to set lamp two to sixty four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":64}` |
| UC04-BRI-B2-065-A | 1 | UC-04 | set lamp two to sixty five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":65}` |
| UC04-BRI-B2-065-A-N1 | 2 | UC-04 | could you set lamp two to sixty five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":65}` |
| UC04-BRI-B2-065-A-N2 | 2 | UC-04 | I would like you to set lamp two to sixty five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":65}` |
| UC04-BRI-B2-066-A | 1 | UC-04 | set lamp two to sixty six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":66}` |
| UC04-BRI-B2-066-A-N1 | 2 | UC-04 | could you set lamp two to sixty six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":66}` |
| UC04-BRI-B2-066-A-N2 | 2 | UC-04 | I would like you to set lamp two to sixty six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":66}` |
| UC04-BRI-B2-067-A | 1 | UC-04 | set lamp two to sixty seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":67}` |
| UC04-BRI-B2-067-A-N1 | 2 | UC-04 | could you set lamp two to sixty seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":67}` |
| UC04-BRI-B2-067-A-N2 | 2 | UC-04 | I would like you to set lamp two to sixty seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":67}` |
| UC04-BRI-B2-068-A | 1 | UC-04 | set lamp two to sixty eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":68}` |
| UC04-BRI-B2-068-A-N1 | 2 | UC-04 | could you set lamp two to sixty eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":68}` |
| UC04-BRI-B2-068-A-N2 | 2 | UC-04 | I would like you to set lamp two to sixty eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":68}` |
| UC04-BRI-B2-069-A | 1 | UC-04 | set lamp two to sixty nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":69}` |
| UC04-BRI-B2-069-A-N1 | 2 | UC-04 | could you set lamp two to sixty nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":69}` |
| UC04-BRI-B2-069-A-N2 | 2 | UC-04 | I would like you to set lamp two to sixty nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":69}` |
| UC04-BRI-B2-070-A | 1 | UC-04 | set lamp two to seventy percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":70}` |
| UC04-BRI-B2-070-A-N1 | 2 | UC-04 | could you set lamp two to seventy percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":70}` |
| UC04-BRI-B2-070-A-N2 | 2 | UC-04 | I would like you to set lamp two to seventy percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":70}` |
| UC04-BRI-B2-071-A | 1 | UC-04 | set lamp two to seventy one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":71}` |
| UC04-BRI-B2-071-A-N1 | 2 | UC-04 | could you set lamp two to seventy one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":71}` |
| UC04-BRI-B2-071-A-N2 | 2 | UC-04 | I would like you to set lamp two to seventy one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":71}` |
| UC04-BRI-B2-072-A | 1 | UC-04 | set lamp two to seventy two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":72}` |
| UC04-BRI-B2-072-A-N1 | 2 | UC-04 | could you set lamp two to seventy two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":72}` |
| UC04-BRI-B2-072-A-N2 | 2 | UC-04 | I would like you to set lamp two to seventy two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":72}` |
| UC04-BRI-B2-073-A | 1 | UC-04 | set lamp two to seventy three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":73}` |
| UC04-BRI-B2-073-A-N1 | 2 | UC-04 | could you set lamp two to seventy three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":73}` |
| UC04-BRI-B2-073-A-N2 | 2 | UC-04 | I would like you to set lamp two to seventy three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":73}` |
| UC04-BRI-B2-074-A | 1 | UC-04 | set lamp two to seventy four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":74}` |
| UC04-BRI-B2-074-A-N1 | 2 | UC-04 | could you set lamp two to seventy four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":74}` |
| UC04-BRI-B2-074-A-N2 | 2 | UC-04 | I would like you to set lamp two to seventy four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":74}` |
| UC04-BRI-B2-075-A | 1 | UC-04 | set lamp two to seventy five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":75}` |
| UC04-BRI-B2-075-A-N1 | 2 | UC-04 | could you set lamp two to seventy five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":75}` |
| UC04-BRI-B2-075-A-N2 | 2 | UC-04 | I would like you to set lamp two to seventy five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":75}` |
| UC04-BRI-B2-076-A | 1 | UC-04 | set lamp two to seventy six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":76}` |
| UC04-BRI-B2-076-A-N1 | 2 | UC-04 | could you set lamp two to seventy six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":76}` |
| UC04-BRI-B2-076-A-N2 | 2 | UC-04 | I would like you to set lamp two to seventy six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":76}` |
| UC04-BRI-B2-077-A | 1 | UC-04 | set lamp two to seventy seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":77}` |
| UC04-BRI-B2-077-A-N1 | 2 | UC-04 | could you set lamp two to seventy seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":77}` |
| UC04-BRI-B2-077-A-N2 | 2 | UC-04 | I would like you to set lamp two to seventy seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":77}` |
| UC04-BRI-B2-078-A | 1 | UC-04 | set lamp two to seventy eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":78}` |
| UC04-BRI-B2-078-A-N1 | 2 | UC-04 | could you set lamp two to seventy eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":78}` |
| UC04-BRI-B2-078-A-N2 | 2 | UC-04 | I would like you to set lamp two to seventy eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":78}` |
| UC04-BRI-B2-079-A | 1 | UC-04 | set lamp two to seventy nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":79}` |
| UC04-BRI-B2-079-A-N1 | 2 | UC-04 | could you set lamp two to seventy nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":79}` |
| UC04-BRI-B2-079-A-N2 | 2 | UC-04 | I would like you to set lamp two to seventy nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":79}` |
| UC04-BRI-B2-080-A | 1 | UC-04 | set lamp two to eighty percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":80}` |
| UC04-BRI-B2-080-A-N1 | 2 | UC-04 | could you set lamp two to eighty percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":80}` |
| UC04-BRI-B2-080-A-N2 | 2 | UC-04 | I would like you to set lamp two to eighty percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":80}` |
| UC04-BRI-B2-081-A | 1 | UC-04 | set lamp two to eighty one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":81}` |
| UC04-BRI-B2-081-A-N1 | 2 | UC-04 | could you set lamp two to eighty one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":81}` |
| UC04-BRI-B2-081-A-N2 | 2 | UC-04 | I would like you to set lamp two to eighty one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":81}` |
| UC04-BRI-B2-082-A | 1 | UC-04 | set lamp two to eighty two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":82}` |
| UC04-BRI-B2-082-A-N1 | 2 | UC-04 | could you set lamp two to eighty two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":82}` |
| UC04-BRI-B2-082-A-N2 | 2 | UC-04 | I would like you to set lamp two to eighty two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":82}` |
| UC04-BRI-B2-083-A | 1 | UC-04 | set lamp two to eighty three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":83}` |
| UC04-BRI-B2-083-A-N1 | 2 | UC-04 | could you set lamp two to eighty three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":83}` |
| UC04-BRI-B2-083-A-N2 | 2 | UC-04 | I would like you to set lamp two to eighty three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":83}` |
| UC04-BRI-B2-084-A | 1 | UC-04 | set lamp two to eighty four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":84}` |
| UC04-BRI-B2-084-A-N1 | 2 | UC-04 | could you set lamp two to eighty four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":84}` |
| UC04-BRI-B2-084-A-N2 | 2 | UC-04 | I would like you to set lamp two to eighty four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":84}` |
| UC04-BRI-B2-085-A | 1 | UC-04 | set lamp two to eighty five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":85}` |
| UC04-BRI-B2-085-A-N1 | 2 | UC-04 | could you set lamp two to eighty five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":85}` |
| UC04-BRI-B2-085-A-N2 | 2 | UC-04 | I would like you to set lamp two to eighty five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":85}` |
| UC04-BRI-B2-086-A | 1 | UC-04 | set lamp two to eighty six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":86}` |
| UC04-BRI-B2-086-A-N1 | 2 | UC-04 | could you set lamp two to eighty six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":86}` |
| UC04-BRI-B2-086-A-N2 | 2 | UC-04 | I would like you to set lamp two to eighty six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":86}` |
| UC04-BRI-B2-087-A | 1 | UC-04 | set lamp two to eighty seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":87}` |
| UC04-BRI-B2-087-A-N1 | 2 | UC-04 | could you set lamp two to eighty seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":87}` |
| UC04-BRI-B2-087-A-N2 | 2 | UC-04 | I would like you to set lamp two to eighty seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":87}` |
| UC04-BRI-B2-088-A | 1 | UC-04 | set lamp two to eighty eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":88}` |
| UC04-BRI-B2-088-A-N1 | 2 | UC-04 | could you set lamp two to eighty eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":88}` |
| UC04-BRI-B2-088-A-N2 | 2 | UC-04 | I would like you to set lamp two to eighty eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":88}` |
| UC04-BRI-B2-089-A | 1 | UC-04 | set lamp two to eighty nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":89}` |
| UC04-BRI-B2-089-A-N1 | 2 | UC-04 | could you set lamp two to eighty nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":89}` |
| UC04-BRI-B2-089-A-N2 | 2 | UC-04 | I would like you to set lamp two to eighty nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":89}` |
| UC04-BRI-B2-090-A | 1 | UC-04 | set lamp two to ninety percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":90}` |
| UC04-BRI-B2-090-A-N1 | 2 | UC-04 | could you set lamp two to ninety percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":90}` |
| UC04-BRI-B2-090-A-N2 | 2 | UC-04 | I would like you to set lamp two to ninety percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":90}` |
| UC04-BRI-B2-091-A | 1 | UC-04 | set lamp two to ninety one percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":91}` |
| UC04-BRI-B2-091-A-N1 | 2 | UC-04 | could you set lamp two to ninety one percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":91}` |
| UC04-BRI-B2-091-A-N2 | 2 | UC-04 | I would like you to set lamp two to ninety one percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":91}` |
| UC04-BRI-B2-092-A | 1 | UC-04 | set lamp two to ninety two percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":92}` |
| UC04-BRI-B2-092-A-N1 | 2 | UC-04 | could you set lamp two to ninety two percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":92}` |
| UC04-BRI-B2-092-A-N2 | 2 | UC-04 | I would like you to set lamp two to ninety two percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":92}` |
| UC04-BRI-B2-093-A | 1 | UC-04 | set lamp two to ninety three percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":93}` |
| UC04-BRI-B2-093-A-N1 | 2 | UC-04 | could you set lamp two to ninety three percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":93}` |
| UC04-BRI-B2-093-A-N2 | 2 | UC-04 | I would like you to set lamp two to ninety three percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":93}` |
| UC04-BRI-B2-094-A | 1 | UC-04 | set lamp two to ninety four percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":94}` |
| UC04-BRI-B2-094-A-N1 | 2 | UC-04 | could you set lamp two to ninety four percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":94}` |
| UC04-BRI-B2-094-A-N2 | 2 | UC-04 | I would like you to set lamp two to ninety four percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":94}` |
| UC04-BRI-B2-095-A | 1 | UC-04 | set lamp two to ninety five percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":95}` |
| UC04-BRI-B2-095-A-N1 | 2 | UC-04 | could you set lamp two to ninety five percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":95}` |
| UC04-BRI-B2-095-A-N2 | 2 | UC-04 | I would like you to set lamp two to ninety five percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":95}` |
| UC04-BRI-B2-096-A | 1 | UC-04 | set lamp two to ninety six percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":96}` |
| UC04-BRI-B2-096-A-N1 | 2 | UC-04 | could you set lamp two to ninety six percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":96}` |
| UC04-BRI-B2-096-A-N2 | 2 | UC-04 | I would like you to set lamp two to ninety six percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":96}` |
| UC04-BRI-B2-097-A | 1 | UC-04 | set lamp two to ninety seven percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":97}` |
| UC04-BRI-B2-097-A-N1 | 2 | UC-04 | could you set lamp two to ninety seven percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":97}` |
| UC04-BRI-B2-097-A-N2 | 2 | UC-04 | I would like you to set lamp two to ninety seven percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":97}` |
| UC04-BRI-B2-098-A | 1 | UC-04 | set lamp two to ninety eight percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":98}` |
| UC04-BRI-B2-098-A-N1 | 2 | UC-04 | could you set lamp two to ninety eight percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":98}` |
| UC04-BRI-B2-098-A-N2 | 2 | UC-04 | I would like you to set lamp two to ninety eight percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":98}` |
| UC04-BRI-B2-099-A | 1 | UC-04 | set lamp two to ninety nine percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":99}` |
| UC04-BRI-B2-099-A-N1 | 2 | UC-04 | could you set lamp two to ninety nine percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":99}` |
| UC04-BRI-B2-099-A-N2 | 2 | UC-04 | I would like you to set lamp two to ninety nine percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":99}` |
| UC04-BRI-B2-100-A | 1 | UC-04 | set lamp two to one hundred percent | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":100}` |
| UC04-BRI-B2-100-A-N1 | 2 | UC-04 | could you set lamp two to one hundred percent, please | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":100}` |
| UC04-BRI-B2-100-A-N2 | 2 | UC-04 | I would like you to set lamp two to one hundred percent for me | `{"entity_id":"light.bedroom","intent":"BRIGHTNESS","value":100}` |
| UC04-DELTA-B2-P05 | 1 | UC-04 | increase lamp two by five percent | `{"delta":5,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B2-P05-N1 | 2 | UC-04 | could you increase lamp two by five percent, please | `{"delta":5,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B2-P05-N2 | 2 | UC-04 | I would like you to increase lamp two by five percent for me | `{"delta":5,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B2-P10 | 1 | UC-04 | increase lamp two by ten percent | `{"delta":10,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B2-P10-N1 | 2 | UC-04 | could you increase lamp two by ten percent, please | `{"delta":10,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B2-P10-N2 | 2 | UC-04 | I would like you to increase lamp two by ten percent for me | `{"delta":10,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B2-M05 | 1 | UC-04 | decrease lamp two by five percent | `{"delta":-5,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B2-M05-N1 | 2 | UC-04 | could you decrease lamp two by five percent, please | `{"delta":-5,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B2-M05-N2 | 2 | UC-04 | I would like you to decrease lamp two by five percent for me | `{"delta":-5,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B2-M10 | 1 | UC-04 | decrease lamp two by ten percent | `{"delta":-10,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B2-M10-N1 | 2 | UC-04 | could you decrease lamp two by ten percent, please | `{"delta":-10,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-DELTA-B2-M10-N2 | 2 | UC-04 | I would like you to decrease lamp two by ten percent for me | `{"delta":-10,"entity_id":"light.bedroom","intent":"BRIGHTNESS_DELTA"}` |
| UC04-COLOR-K-RED | 1 | UC-04 | set kitchen lamp to red | `{"color":"red","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-RED-N1 | 2 | UC-04 | could you set kitchen lamp to red, please | `{"color":"red","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-RED-N2 | 2 | UC-04 | I would like you to set kitchen lamp to red for me | `{"color":"red","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-GREEN | 1 | UC-04 | set kitchen lamp to green | `{"color":"green","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-GREEN-N1 | 2 | UC-04 | could you set kitchen lamp to green, please | `{"color":"green","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-GREEN-N2 | 2 | UC-04 | I would like you to set kitchen lamp to green for me | `{"color":"green","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-BLUE | 1 | UC-04 | set kitchen lamp to blue | `{"color":"blue","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-BLUE-N1 | 2 | UC-04 | could you set kitchen lamp to blue, please | `{"color":"blue","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-BLUE-N2 | 2 | UC-04 | I would like you to set kitchen lamp to blue for me | `{"color":"blue","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-WHITE | 1 | UC-04 | set kitchen lamp to white | `{"color":"white","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-WHITE-N1 | 2 | UC-04 | could you set kitchen lamp to white, please | `{"color":"white","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-WHITE-N2 | 2 | UC-04 | I would like you to set kitchen lamp to white for me | `{"color":"white","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-YELLOW | 1 | UC-04 | set kitchen lamp to yellow | `{"color":"yellow","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-YELLOW-N1 | 2 | UC-04 | could you set kitchen lamp to yellow, please | `{"color":"yellow","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-YELLOW-N2 | 2 | UC-04 | I would like you to set kitchen lamp to yellow for me | `{"color":"yellow","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-WARM_WHITE | 1 | UC-04 | set kitchen lamp to warm white | `{"color":"warm_white","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-WARM_WHITE-N1 | 2 | UC-04 | could you set kitchen lamp to warm white, please | `{"color":"warm_white","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC04-COLOR-K-WARM_WHITE-N2 | 2 | UC-04 | I would like you to set kitchen lamp to warm white for me | `{"color":"warm_white","entity_id":"light.kitchen","intent":"COLOR_SET"}` |
| UC05-SET-0060-A | 1 | UC-05 | set a timer for one minute | `{"duration_seconds":60,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0060-A-N1 | 2 | UC-05 | could you set a timer for one minute, please | `{"duration_seconds":60,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0060-A-N2 | 2 | UC-05 | I would like you to set a timer for one minute for me | `{"duration_seconds":60,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0060-B | 1 | UC-05 | start a timer for one minute | `{"duration_seconds":60,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0060-B-N1 | 2 | UC-05 | could you start a timer for one minute, please | `{"duration_seconds":60,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0060-B-N2 | 2 | UC-05 | I would like you to start a timer for one minute for me | `{"duration_seconds":60,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0120-A | 1 | UC-05 | set a timer for two minutes | `{"duration_seconds":120,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0120-A-N1 | 2 | UC-05 | could you set a timer for two minutes, please | `{"duration_seconds":120,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0120-A-N2 | 2 | UC-05 | I would like you to set a timer for two minutes for me | `{"duration_seconds":120,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0120-B | 1 | UC-05 | start a timer for two minutes | `{"duration_seconds":120,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0120-B-N1 | 2 | UC-05 | could you start a timer for two minutes, please | `{"duration_seconds":120,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0120-B-N2 | 2 | UC-05 | I would like you to start a timer for two minutes for me | `{"duration_seconds":120,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0300-A | 1 | UC-05 | set a timer for five minutes | `{"duration_seconds":300,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0300-A-N1 | 2 | UC-05 | could you set a timer for five minutes, please | `{"duration_seconds":300,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0300-A-N2 | 2 | UC-05 | I would like you to set a timer for five minutes for me | `{"duration_seconds":300,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0300-B | 1 | UC-05 | start a timer for five minutes | `{"duration_seconds":300,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0300-B-N1 | 2 | UC-05 | could you start a timer for five minutes, please | `{"duration_seconds":300,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0300-B-N2 | 2 | UC-05 | I would like you to start a timer for five minutes for me | `{"duration_seconds":300,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0600-A | 1 | UC-05 | set a timer for ten minutes | `{"duration_seconds":600,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0600-A-N1 | 2 | UC-05 | could you set a timer for ten minutes, please | `{"duration_seconds":600,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0600-A-N2 | 2 | UC-05 | I would like you to set a timer for ten minutes for me | `{"duration_seconds":600,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0600-B | 1 | UC-05 | start a timer for ten minutes | `{"duration_seconds":600,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0600-B-N1 | 2 | UC-05 | could you start a timer for ten minutes, please | `{"duration_seconds":600,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0600-B-N2 | 2 | UC-05 | I would like you to start a timer for ten minutes for me | `{"duration_seconds":600,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0900-A | 1 | UC-05 | set a timer for fifteen minutes | `{"duration_seconds":900,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0900-A-N1 | 2 | UC-05 | could you set a timer for fifteen minutes, please | `{"duration_seconds":900,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0900-A-N2 | 2 | UC-05 | I would like you to set a timer for fifteen minutes for me | `{"duration_seconds":900,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0900-B | 1 | UC-05 | start a timer for fifteen minutes | `{"duration_seconds":900,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0900-B-N1 | 2 | UC-05 | could you start a timer for fifteen minutes, please | `{"duration_seconds":900,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-0900-B-N2 | 2 | UC-05 | I would like you to start a timer for fifteen minutes for me | `{"duration_seconds":900,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-1800-A | 1 | UC-05 | set a timer for thirty minutes | `{"duration_seconds":1800,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-1800-A-N1 | 2 | UC-05 | could you set a timer for thirty minutes, please | `{"duration_seconds":1800,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-1800-A-N2 | 2 | UC-05 | I would like you to set a timer for thirty minutes for me | `{"duration_seconds":1800,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-1800-B | 1 | UC-05 | start a timer for thirty minutes | `{"duration_seconds":1800,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-1800-B-N1 | 2 | UC-05 | could you start a timer for thirty minutes, please | `{"duration_seconds":1800,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-1800-B-N2 | 2 | UC-05 | I would like you to start a timer for thirty minutes for me | `{"duration_seconds":1800,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-3600-A | 1 | UC-05 | set a timer for one hour | `{"duration_seconds":3600,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-3600-A-N1 | 2 | UC-05 | could you set a timer for one hour, please | `{"duration_seconds":3600,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-3600-A-N2 | 2 | UC-05 | I would like you to set a timer for one hour for me | `{"duration_seconds":3600,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-3600-B | 1 | UC-05 | start a timer for one hour | `{"duration_seconds":3600,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-3600-B-N1 | 2 | UC-05 | could you start a timer for one hour, please | `{"duration_seconds":3600,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-SET-3600-B-N2 | 2 | UC-05 | I would like you to start a timer for one hour for me | `{"duration_seconds":3600,"intent":"TIMER_SET","timer_id":"timer.primary"}` |
| UC05-CANCEL-A | 1 | UC-05 | cancel timer | `{"intent":"TIMER_CANCEL","timer_id":"timer.primary"}` |
| UC05-CANCEL-A-N1 | 2 | UC-05 | could you cancel timer, please | `{"intent":"TIMER_CANCEL","timer_id":"timer.primary"}` |
| UC05-CANCEL-A-N2 | 2 | UC-05 | I would like you to cancel timer for me | `{"intent":"TIMER_CANCEL","timer_id":"timer.primary"}` |
| UC05-CANCEL-B | 1 | UC-05 | stop timer | `{"intent":"TIMER_CANCEL","timer_id":"timer.primary"}` |
| UC05-CANCEL-B-N1 | 2 | UC-05 | could you stop timer, please | `{"intent":"TIMER_CANCEL","timer_id":"timer.primary"}` |
| UC05-CANCEL-B-N2 | 2 | UC-05 | I would like you to stop timer for me | `{"intent":"TIMER_CANCEL","timer_id":"timer.primary"}` |
| UC06-SET-0600-A | 1 | UC-06 | set an alarm for six a m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:00","timezone":"Asia/Manila"}` |
| UC06-SET-0600-A-N1 | 2 | UC-06 | could you set an alarm for six a m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:00","timezone":"Asia/Manila"}` |
| UC06-SET-0600-A-N2 | 2 | UC-06 | I would like you to set an alarm for six a m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:00","timezone":"Asia/Manila"}` |
| UC06-SET-0600-B | 1 | UC-06 | wake me at six a m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:00","timezone":"Asia/Manila"}` |
| UC06-SET-0600-B-N1 | 2 | UC-06 | could you wake me at six a m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:00","timezone":"Asia/Manila"}` |
| UC06-SET-0600-B-N2 | 2 | UC-06 | I would like you to wake me at six a m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:00","timezone":"Asia/Manila"}` |
| UC06-SET-0630-A | 1 | UC-06 | set an alarm for six thirty a m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:30","timezone":"Asia/Manila"}` |
| UC06-SET-0630-A-N1 | 2 | UC-06 | could you set an alarm for six thirty a m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:30","timezone":"Asia/Manila"}` |
| UC06-SET-0630-A-N2 | 2 | UC-06 | I would like you to set an alarm for six thirty a m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:30","timezone":"Asia/Manila"}` |
| UC06-SET-0630-B | 1 | UC-06 | wake me at six thirty a m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:30","timezone":"Asia/Manila"}` |
| UC06-SET-0630-B-N1 | 2 | UC-06 | could you wake me at six thirty a m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:30","timezone":"Asia/Manila"}` |
| UC06-SET-0630-B-N2 | 2 | UC-06 | I would like you to wake me at six thirty a m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"06:30","timezone":"Asia/Manila"}` |
| UC06-SET-0700-A | 1 | UC-06 | set an alarm for seven a m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"07:00","timezone":"Asia/Manila"}` |
| UC06-SET-0700-A-N1 | 2 | UC-06 | could you set an alarm for seven a m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"07:00","timezone":"Asia/Manila"}` |
| UC06-SET-0700-A-N2 | 2 | UC-06 | I would like you to set an alarm for seven a m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"07:00","timezone":"Asia/Manila"}` |
| UC06-SET-0700-B | 1 | UC-06 | wake me at seven a m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"07:00","timezone":"Asia/Manila"}` |
| UC06-SET-0700-B-N1 | 2 | UC-06 | could you wake me at seven a m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"07:00","timezone":"Asia/Manila"}` |
| UC06-SET-0700-B-N2 | 2 | UC-06 | I would like you to wake me at seven a m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"07:00","timezone":"Asia/Manila"}` |
| UC06-SET-0730-A | 1 | UC-06 | set an alarm for seven thirty a m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"07:30","timezone":"Asia/Manila"}` |
| UC06-SET-0730-A-N1 | 2 | UC-06 | could you set an alarm for seven thirty a m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"07:30","timezone":"Asia/Manila"}` |
| UC06-SET-0730-A-N2 | 2 | UC-06 | I would like you to set an alarm for seven thirty a m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"07:30","timezone":"Asia/Manila"}` |
| UC06-SET-0730-B | 1 | UC-06 | wake me at seven thirty a m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"07:30","timezone":"Asia/Manila"}` |
| UC06-SET-0730-B-N1 | 2 | UC-06 | could you wake me at seven thirty a m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"07:30","timezone":"Asia/Manila"}` |
| UC06-SET-0730-B-N2 | 2 | UC-06 | I would like you to wake me at seven thirty a m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"07:30","timezone":"Asia/Manila"}` |
| UC06-SET-0800-A | 1 | UC-06 | set an alarm for eight a m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"08:00","timezone":"Asia/Manila"}` |
| UC06-SET-0800-A-N1 | 2 | UC-06 | could you set an alarm for eight a m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"08:00","timezone":"Asia/Manila"}` |
| UC06-SET-0800-A-N2 | 2 | UC-06 | I would like you to set an alarm for eight a m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"08:00","timezone":"Asia/Manila"}` |
| UC06-SET-0800-B | 1 | UC-06 | wake me at eight a m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"08:00","timezone":"Asia/Manila"}` |
| UC06-SET-0800-B-N1 | 2 | UC-06 | could you wake me at eight a m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"08:00","timezone":"Asia/Manila"}` |
| UC06-SET-0800-B-N2 | 2 | UC-06 | I would like you to wake me at eight a m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"08:00","timezone":"Asia/Manila"}` |
| UC06-SET-0830-A | 1 | UC-06 | set an alarm for eight thirty a m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"08:30","timezone":"Asia/Manila"}` |
| UC06-SET-0830-A-N1 | 2 | UC-06 | could you set an alarm for eight thirty a m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"08:30","timezone":"Asia/Manila"}` |
| UC06-SET-0830-A-N2 | 2 | UC-06 | I would like you to set an alarm for eight thirty a m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"08:30","timezone":"Asia/Manila"}` |
| UC06-SET-0830-B | 1 | UC-06 | wake me at eight thirty a m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"08:30","timezone":"Asia/Manila"}` |
| UC06-SET-0830-B-N1 | 2 | UC-06 | could you wake me at eight thirty a m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"08:30","timezone":"Asia/Manila"}` |
| UC06-SET-0830-B-N2 | 2 | UC-06 | I would like you to wake me at eight thirty a m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"08:30","timezone":"Asia/Manila"}` |
| UC06-SET-1800-A | 1 | UC-06 | set an alarm for six p m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"18:00","timezone":"Asia/Manila"}` |
| UC06-SET-1800-A-N1 | 2 | UC-06 | could you set an alarm for six p m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"18:00","timezone":"Asia/Manila"}` |
| UC06-SET-1800-A-N2 | 2 | UC-06 | I would like you to set an alarm for six p m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"18:00","timezone":"Asia/Manila"}` |
| UC06-SET-1800-B | 1 | UC-06 | wake me at six p m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"18:00","timezone":"Asia/Manila"}` |
| UC06-SET-1800-B-N1 | 2 | UC-06 | could you wake me at six p m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"18:00","timezone":"Asia/Manila"}` |
| UC06-SET-1800-B-N2 | 2 | UC-06 | I would like you to wake me at six p m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"18:00","timezone":"Asia/Manila"}` |
| UC06-SET-1830-A | 1 | UC-06 | set an alarm for six thirty p m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"18:30","timezone":"Asia/Manila"}` |
| UC06-SET-1830-A-N1 | 2 | UC-06 | could you set an alarm for six thirty p m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"18:30","timezone":"Asia/Manila"}` |
| UC06-SET-1830-A-N2 | 2 | UC-06 | I would like you to set an alarm for six thirty p m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"18:30","timezone":"Asia/Manila"}` |
| UC06-SET-1830-B | 1 | UC-06 | wake me at six thirty p m | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"18:30","timezone":"Asia/Manila"}` |
| UC06-SET-1830-B-N1 | 2 | UC-06 | could you wake me at six thirty p m, please | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"18:30","timezone":"Asia/Manila"}` |
| UC06-SET-1830-B-N2 | 2 | UC-06 | I would like you to wake me at six thirty p m for me | `{"alarm_id":"alarm.primary","intent":"ALARM_SET","repeat":"once","time_local":"18:30","timezone":"Asia/Manila"}` |
| UC06-CANCEL-A | 1 | UC-06 | cancel alarm | `{"alarm_id":"alarm.primary","intent":"ALARM_CANCEL"}` |
| UC06-CANCEL-A-N1 | 2 | UC-06 | could you cancel alarm, please | `{"alarm_id":"alarm.primary","intent":"ALARM_CANCEL"}` |
| UC06-CANCEL-A-N2 | 2 | UC-06 | I would like you to cancel alarm for me | `{"alarm_id":"alarm.primary","intent":"ALARM_CANCEL"}` |
| UC06-CANCEL-B | 1 | UC-06 | delete alarm | `{"alarm_id":"alarm.primary","intent":"ALARM_CANCEL"}` |
| UC06-CANCEL-B-N1 | 2 | UC-06 | could you delete alarm, please | `{"alarm_id":"alarm.primary","intent":"ALARM_CANCEL"}` |
| UC06-CANCEL-B-N2 | 2 | UC-06 | I would like you to delete alarm for me | `{"alarm_id":"alarm.primary","intent":"ALARM_CANCEL"}` |
| UC07-ABS-16 | 1 | UC-07 | set thermostat to sixteen degrees celsius | `{"celsius":16,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-16-N1 | 2 | UC-07 | could you set thermostat to sixteen degrees celsius, please | `{"celsius":16,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-16-N2 | 2 | UC-07 | I would like you to set thermostat to sixteen degrees celsius for me | `{"celsius":16,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-17 | 1 | UC-07 | set thermostat to seventeen degrees celsius | `{"celsius":17,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-17-N1 | 2 | UC-07 | could you set thermostat to seventeen degrees celsius, please | `{"celsius":17,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-17-N2 | 2 | UC-07 | I would like you to set thermostat to seventeen degrees celsius for me | `{"celsius":17,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-18 | 1 | UC-07 | set thermostat to eighteen degrees celsius | `{"celsius":18,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-18-N1 | 2 | UC-07 | could you set thermostat to eighteen degrees celsius, please | `{"celsius":18,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-18-N2 | 2 | UC-07 | I would like you to set thermostat to eighteen degrees celsius for me | `{"celsius":18,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-19 | 1 | UC-07 | set thermostat to nineteen degrees celsius | `{"celsius":19,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-19-N1 | 2 | UC-07 | could you set thermostat to nineteen degrees celsius, please | `{"celsius":19,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-19-N2 | 2 | UC-07 | I would like you to set thermostat to nineteen degrees celsius for me | `{"celsius":19,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-20 | 1 | UC-07 | set thermostat to twenty degrees celsius | `{"celsius":20,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-20-N1 | 2 | UC-07 | could you set thermostat to twenty degrees celsius, please | `{"celsius":20,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-20-N2 | 2 | UC-07 | I would like you to set thermostat to twenty degrees celsius for me | `{"celsius":20,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-21 | 1 | UC-07 | set thermostat to twenty one degrees celsius | `{"celsius":21,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-21-N1 | 2 | UC-07 | could you set thermostat to twenty one degrees celsius, please | `{"celsius":21,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-21-N2 | 2 | UC-07 | I would like you to set thermostat to twenty one degrees celsius for me | `{"celsius":21,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-22 | 1 | UC-07 | set thermostat to twenty two degrees celsius | `{"celsius":22,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-22-N1 | 2 | UC-07 | could you set thermostat to twenty two degrees celsius, please | `{"celsius":22,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-22-N2 | 2 | UC-07 | I would like you to set thermostat to twenty two degrees celsius for me | `{"celsius":22,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-23 | 1 | UC-07 | set thermostat to twenty three degrees celsius | `{"celsius":23,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-23-N1 | 2 | UC-07 | could you set thermostat to twenty three degrees celsius, please | `{"celsius":23,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-23-N2 | 2 | UC-07 | I would like you to set thermostat to twenty three degrees celsius for me | `{"celsius":23,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-24 | 1 | UC-07 | set thermostat to twenty four degrees celsius | `{"celsius":24,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-24-N1 | 2 | UC-07 | could you set thermostat to twenty four degrees celsius, please | `{"celsius":24,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-24-N2 | 2 | UC-07 | I would like you to set thermostat to twenty four degrees celsius for me | `{"celsius":24,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-25 | 1 | UC-07 | set thermostat to twenty five degrees celsius | `{"celsius":25,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-25-N1 | 2 | UC-07 | could you set thermostat to twenty five degrees celsius, please | `{"celsius":25,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-25-N2 | 2 | UC-07 | I would like you to set thermostat to twenty five degrees celsius for me | `{"celsius":25,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-26 | 1 | UC-07 | set thermostat to twenty six degrees celsius | `{"celsius":26,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-26-N1 | 2 | UC-07 | could you set thermostat to twenty six degrees celsius, please | `{"celsius":26,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-26-N2 | 2 | UC-07 | I would like you to set thermostat to twenty six degrees celsius for me | `{"celsius":26,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-27 | 1 | UC-07 | set thermostat to twenty seven degrees celsius | `{"celsius":27,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-27-N1 | 2 | UC-07 | could you set thermostat to twenty seven degrees celsius, please | `{"celsius":27,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-27-N2 | 2 | UC-07 | I would like you to set thermostat to twenty seven degrees celsius for me | `{"celsius":27,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-28 | 1 | UC-07 | set thermostat to twenty eight degrees celsius | `{"celsius":28,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-28-N1 | 2 | UC-07 | could you set thermostat to twenty eight degrees celsius, please | `{"celsius":28,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-28-N2 | 2 | UC-07 | I would like you to set thermostat to twenty eight degrees celsius for me | `{"celsius":28,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-29 | 1 | UC-07 | set thermostat to twenty nine degrees celsius | `{"celsius":29,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-29-N1 | 2 | UC-07 | could you set thermostat to twenty nine degrees celsius, please | `{"celsius":29,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-29-N2 | 2 | UC-07 | I would like you to set thermostat to twenty nine degrees celsius for me | `{"celsius":29,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-30 | 1 | UC-07 | set thermostat to thirty degrees celsius | `{"celsius":30,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-30-N1 | 2 | UC-07 | could you set thermostat to thirty degrees celsius, please | `{"celsius":30,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-ABS-30-N2 | 2 | UC-07 | I would like you to set thermostat to thirty degrees celsius for me | `{"celsius":30,"entity_id":"climate.home","intent":"TEMPERATURE_SET"}` |
| UC07-DELTA-P1 | 1 | UC-07 | increase thermostat by one degree celsius | `{"delta_celsius":1,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC07-DELTA-P1-N1 | 2 | UC-07 | could you increase thermostat by one degree celsius, please | `{"delta_celsius":1,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC07-DELTA-P1-N2 | 2 | UC-07 | I would like you to increase thermostat by one degree celsius for me | `{"delta_celsius":1,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC07-DELTA-P2 | 1 | UC-07 | increase thermostat by two degrees celsius | `{"delta_celsius":2,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC07-DELTA-P2-N1 | 2 | UC-07 | could you increase thermostat by two degrees celsius, please | `{"delta_celsius":2,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC07-DELTA-P2-N2 | 2 | UC-07 | I would like you to increase thermostat by two degrees celsius for me | `{"delta_celsius":2,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC07-DELTA-M1 | 1 | UC-07 | decrease thermostat by one degree celsius | `{"delta_celsius":-1,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC07-DELTA-M1-N1 | 2 | UC-07 | could you decrease thermostat by one degree celsius, please | `{"delta_celsius":-1,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC07-DELTA-M1-N2 | 2 | UC-07 | I would like you to decrease thermostat by one degree celsius for me | `{"delta_celsius":-1,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC07-DELTA-M2 | 1 | UC-07 | decrease thermostat by two degrees celsius | `{"delta_celsius":-2,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC07-DELTA-M2-N1 | 2 | UC-07 | could you decrease thermostat by two degrees celsius, please | `{"delta_celsius":-2,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC07-DELTA-M2-N2 | 2 | UC-07 | I would like you to decrease thermostat by two degrees celsius for me | `{"delta_celsius":-2,"entity_id":"climate.home","intent":"TEMPERATURE_DELTA"}` |
| UC08-PAUSE-A | 1 | UC-08 | pause music | `{"entity_id":"media.main","intent":"MEDIA_PAUSE"}` |
| UC08-PAUSE-A-N1 | 2 | UC-08 | could you pause music, please | `{"entity_id":"media.main","intent":"MEDIA_PAUSE"}` |
| UC08-PAUSE-A-N2 | 2 | UC-08 | I would like you to pause music for me | `{"entity_id":"media.main","intent":"MEDIA_PAUSE"}` |
| UC08-PAUSE-B | 1 | UC-08 | pause playback | `{"entity_id":"media.main","intent":"MEDIA_PAUSE"}` |
| UC08-PAUSE-B-N1 | 2 | UC-08 | could you pause playback, please | `{"entity_id":"media.main","intent":"MEDIA_PAUSE"}` |
| UC08-PAUSE-B-N2 | 2 | UC-08 | I would like you to pause playback for me | `{"entity_id":"media.main","intent":"MEDIA_PAUSE"}` |
| UC08-RESUME-A | 1 | UC-08 | resume music | `{"entity_id":"media.main","intent":"MEDIA_RESUME"}` |
| UC08-RESUME-A-N1 | 2 | UC-08 | could you resume music, please | `{"entity_id":"media.main","intent":"MEDIA_RESUME"}` |
| UC08-RESUME-A-N2 | 2 | UC-08 | I would like you to resume music for me | `{"entity_id":"media.main","intent":"MEDIA_RESUME"}` |
| UC08-RESUME-B | 1 | UC-08 | resume playback | `{"entity_id":"media.main","intent":"MEDIA_RESUME"}` |
| UC08-RESUME-B-N1 | 2 | UC-08 | could you resume playback, please | `{"entity_id":"media.main","intent":"MEDIA_RESUME"}` |
| UC08-RESUME-B-N2 | 2 | UC-08 | I would like you to resume playback for me | `{"entity_id":"media.main","intent":"MEDIA_RESUME"}` |
| UC08-STOP-A | 1 | UC-08 | stop music | `{"entity_id":"media.main","intent":"MEDIA_STOP"}` |
| UC08-STOP-A-N1 | 2 | UC-08 | could you stop music, please | `{"entity_id":"media.main","intent":"MEDIA_STOP"}` |
| UC08-STOP-A-N2 | 2 | UC-08 | I would like you to stop music for me | `{"entity_id":"media.main","intent":"MEDIA_STOP"}` |
| UC08-STOP-B | 1 | UC-08 | stop playback | `{"entity_id":"media.main","intent":"MEDIA_STOP"}` |
| UC08-STOP-B-N1 | 2 | UC-08 | could you stop playback, please | `{"entity_id":"media.main","intent":"MEDIA_STOP"}` |
| UC08-STOP-B-N2 | 2 | UC-08 | I would like you to stop playback for me | `{"entity_id":"media.main","intent":"MEDIA_STOP"}` |
| UC08-NEXT-A | 1 | UC-08 | next track | `{"entity_id":"media.main","intent":"MEDIA_NEXT"}` |
| UC08-NEXT-A-N1 | 2 | UC-08 | could you play the next track, please | `{"entity_id":"media.main","intent":"MEDIA_NEXT"}` |
| UC08-NEXT-A-N2 | 2 | UC-08 | I would like you to play the next track for me | `{"entity_id":"media.main","intent":"MEDIA_NEXT"}` |
| UC08-NEXT-B | 1 | UC-08 | skip track | `{"entity_id":"media.main","intent":"MEDIA_NEXT"}` |
| UC08-NEXT-B-N1 | 2 | UC-08 | could you skip track, please | `{"entity_id":"media.main","intent":"MEDIA_NEXT"}` |
| UC08-NEXT-B-N2 | 2 | UC-08 | I would like you to skip track for me | `{"entity_id":"media.main","intent":"MEDIA_NEXT"}` |
| UC08-PREVIOUS-A | 1 | UC-08 | previous track | `{"entity_id":"media.main","intent":"MEDIA_PREVIOUS"}` |
| UC08-PREVIOUS-A-N1 | 2 | UC-08 | could you play the previous track, please | `{"entity_id":"media.main","intent":"MEDIA_PREVIOUS"}` |
| UC08-PREVIOUS-A-N2 | 2 | UC-08 | I would like you to play the previous track for me | `{"entity_id":"media.main","intent":"MEDIA_PREVIOUS"}` |
| UC08-PREVIOUS-B | 1 | UC-08 | go back a track | `{"entity_id":"media.main","intent":"MEDIA_PREVIOUS"}` |
| UC08-PREVIOUS-B-N1 | 2 | UC-08 | could you go back a track, please | `{"entity_id":"media.main","intent":"MEDIA_PREVIOUS"}` |
| UC08-PREVIOUS-B-N2 | 2 | UC-08 | I would like you to go back a track for me | `{"entity_id":"media.main","intent":"MEDIA_PREVIOUS"}` |
| UC08-MUTE-A | 1 | UC-08 | mute audio | `{"entity_id":"media.main","intent":"MEDIA_MUTE"}` |
| UC08-MUTE-A-N1 | 2 | UC-08 | could you mute audio, please | `{"entity_id":"media.main","intent":"MEDIA_MUTE"}` |
| UC08-MUTE-A-N2 | 2 | UC-08 | I would like you to mute audio for me | `{"entity_id":"media.main","intent":"MEDIA_MUTE"}` |
| UC08-MUTE-B | 1 | UC-08 | mute sound | `{"entity_id":"media.main","intent":"MEDIA_MUTE"}` |
| UC08-MUTE-B-N1 | 2 | UC-08 | could you mute sound, please | `{"entity_id":"media.main","intent":"MEDIA_MUTE"}` |
| UC08-MUTE-B-N2 | 2 | UC-08 | I would like you to mute sound for me | `{"entity_id":"media.main","intent":"MEDIA_MUTE"}` |
| UC08-UNMUTE-A | 1 | UC-08 | unmute audio | `{"entity_id":"media.main","intent":"MEDIA_UNMUTE"}` |
| UC08-UNMUTE-A-N1 | 2 | UC-08 | could you unmute audio, please | `{"entity_id":"media.main","intent":"MEDIA_UNMUTE"}` |
| UC08-UNMUTE-A-N2 | 2 | UC-08 | I would like you to unmute audio for me | `{"entity_id":"media.main","intent":"MEDIA_UNMUTE"}` |
| UC08-UNMUTE-B | 1 | UC-08 | unmute sound | `{"entity_id":"media.main","intent":"MEDIA_UNMUTE"}` |
| UC08-UNMUTE-B-N1 | 2 | UC-08 | could you unmute sound, please | `{"entity_id":"media.main","intent":"MEDIA_UNMUTE"}` |
| UC08-UNMUTE-B-N2 | 2 | UC-08 | I would like you to unmute sound for me | `{"entity_id":"media.main","intent":"MEDIA_UNMUTE"}` |
| UC08-VOL-ABS-000-A | 1 | UC-08 | set volume to zero percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":0}` |
| UC08-VOL-ABS-000-A-N1 | 2 | UC-08 | could you set volume to zero percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":0}` |
| UC08-VOL-ABS-000-A-N2 | 2 | UC-08 | I would like you to set volume to zero percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":0}` |
| UC08-VOL-ABS-001-A | 1 | UC-08 | set volume to one percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":1}` |
| UC08-VOL-ABS-001-A-N1 | 2 | UC-08 | could you set volume to one percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":1}` |
| UC08-VOL-ABS-001-A-N2 | 2 | UC-08 | I would like you to set volume to one percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":1}` |
| UC08-VOL-ABS-002-A | 1 | UC-08 | set volume to two percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":2}` |
| UC08-VOL-ABS-002-A-N1 | 2 | UC-08 | could you set volume to two percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":2}` |
| UC08-VOL-ABS-002-A-N2 | 2 | UC-08 | I would like you to set volume to two percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":2}` |
| UC08-VOL-ABS-003-A | 1 | UC-08 | set volume to three percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":3}` |
| UC08-VOL-ABS-003-A-N1 | 2 | UC-08 | could you set volume to three percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":3}` |
| UC08-VOL-ABS-003-A-N2 | 2 | UC-08 | I would like you to set volume to three percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":3}` |
| UC08-VOL-ABS-004-A | 1 | UC-08 | set volume to four percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":4}` |
| UC08-VOL-ABS-004-A-N1 | 2 | UC-08 | could you set volume to four percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":4}` |
| UC08-VOL-ABS-004-A-N2 | 2 | UC-08 | I would like you to set volume to four percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":4}` |
| UC08-VOL-ABS-005-A | 1 | UC-08 | set volume to five percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":5}` |
| UC08-VOL-ABS-005-A-N1 | 2 | UC-08 | could you set volume to five percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":5}` |
| UC08-VOL-ABS-005-A-N2 | 2 | UC-08 | I would like you to set volume to five percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":5}` |
| UC08-VOL-ABS-006-A | 1 | UC-08 | set volume to six percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":6}` |
| UC08-VOL-ABS-006-A-N1 | 2 | UC-08 | could you set volume to six percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":6}` |
| UC08-VOL-ABS-006-A-N2 | 2 | UC-08 | I would like you to set volume to six percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":6}` |
| UC08-VOL-ABS-007-A | 1 | UC-08 | set volume to seven percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":7}` |
| UC08-VOL-ABS-007-A-N1 | 2 | UC-08 | could you set volume to seven percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":7}` |
| UC08-VOL-ABS-007-A-N2 | 2 | UC-08 | I would like you to set volume to seven percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":7}` |
| UC08-VOL-ABS-008-A | 1 | UC-08 | set volume to eight percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":8}` |
| UC08-VOL-ABS-008-A-N1 | 2 | UC-08 | could you set volume to eight percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":8}` |
| UC08-VOL-ABS-008-A-N2 | 2 | UC-08 | I would like you to set volume to eight percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":8}` |
| UC08-VOL-ABS-009-A | 1 | UC-08 | set volume to nine percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":9}` |
| UC08-VOL-ABS-009-A-N1 | 2 | UC-08 | could you set volume to nine percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":9}` |
| UC08-VOL-ABS-009-A-N2 | 2 | UC-08 | I would like you to set volume to nine percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":9}` |
| UC08-VOL-ABS-010-A | 1 | UC-08 | set volume to ten percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":10}` |
| UC08-VOL-ABS-010-A-N1 | 2 | UC-08 | could you set volume to ten percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":10}` |
| UC08-VOL-ABS-010-A-N2 | 2 | UC-08 | I would like you to set volume to ten percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":10}` |
| UC08-VOL-ABS-011-A | 1 | UC-08 | set volume to eleven percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":11}` |
| UC08-VOL-ABS-011-A-N1 | 2 | UC-08 | could you set volume to eleven percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":11}` |
| UC08-VOL-ABS-011-A-N2 | 2 | UC-08 | I would like you to set volume to eleven percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":11}` |
| UC08-VOL-ABS-012-A | 1 | UC-08 | set volume to twelve percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":12}` |
| UC08-VOL-ABS-012-A-N1 | 2 | UC-08 | could you set volume to twelve percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":12}` |
| UC08-VOL-ABS-012-A-N2 | 2 | UC-08 | I would like you to set volume to twelve percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":12}` |
| UC08-VOL-ABS-013-A | 1 | UC-08 | set volume to thirteen percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":13}` |
| UC08-VOL-ABS-013-A-N1 | 2 | UC-08 | could you set volume to thirteen percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":13}` |
| UC08-VOL-ABS-013-A-N2 | 2 | UC-08 | I would like you to set volume to thirteen percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":13}` |
| UC08-VOL-ABS-014-A | 1 | UC-08 | set volume to fourteen percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":14}` |
| UC08-VOL-ABS-014-A-N1 | 2 | UC-08 | could you set volume to fourteen percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":14}` |
| UC08-VOL-ABS-014-A-N2 | 2 | UC-08 | I would like you to set volume to fourteen percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":14}` |
| UC08-VOL-ABS-015-A | 1 | UC-08 | set volume to fifteen percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":15}` |
| UC08-VOL-ABS-015-A-N1 | 2 | UC-08 | could you set volume to fifteen percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":15}` |
| UC08-VOL-ABS-015-A-N2 | 2 | UC-08 | I would like you to set volume to fifteen percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":15}` |
| UC08-VOL-ABS-016-A | 1 | UC-08 | set volume to sixteen percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":16}` |
| UC08-VOL-ABS-016-A-N1 | 2 | UC-08 | could you set volume to sixteen percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":16}` |
| UC08-VOL-ABS-016-A-N2 | 2 | UC-08 | I would like you to set volume to sixteen percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":16}` |
| UC08-VOL-ABS-017-A | 1 | UC-08 | set volume to seventeen percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":17}` |
| UC08-VOL-ABS-017-A-N1 | 2 | UC-08 | could you set volume to seventeen percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":17}` |
| UC08-VOL-ABS-017-A-N2 | 2 | UC-08 | I would like you to set volume to seventeen percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":17}` |
| UC08-VOL-ABS-018-A | 1 | UC-08 | set volume to eighteen percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":18}` |
| UC08-VOL-ABS-018-A-N1 | 2 | UC-08 | could you set volume to eighteen percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":18}` |
| UC08-VOL-ABS-018-A-N2 | 2 | UC-08 | I would like you to set volume to eighteen percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":18}` |
| UC08-VOL-ABS-019-A | 1 | UC-08 | set volume to nineteen percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":19}` |
| UC08-VOL-ABS-019-A-N1 | 2 | UC-08 | could you set volume to nineteen percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":19}` |
| UC08-VOL-ABS-019-A-N2 | 2 | UC-08 | I would like you to set volume to nineteen percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":19}` |
| UC08-VOL-ABS-020-A | 1 | UC-08 | set volume to twenty percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":20}` |
| UC08-VOL-ABS-020-A-N1 | 2 | UC-08 | could you set volume to twenty percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":20}` |
| UC08-VOL-ABS-020-A-N2 | 2 | UC-08 | I would like you to set volume to twenty percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":20}` |
| UC08-VOL-ABS-021-A | 1 | UC-08 | set volume to twenty one percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":21}` |
| UC08-VOL-ABS-021-A-N1 | 2 | UC-08 | could you set volume to twenty one percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":21}` |
| UC08-VOL-ABS-021-A-N2 | 2 | UC-08 | I would like you to set volume to twenty one percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":21}` |
| UC08-VOL-ABS-022-A | 1 | UC-08 | set volume to twenty two percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":22}` |
| UC08-VOL-ABS-022-A-N1 | 2 | UC-08 | could you set volume to twenty two percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":22}` |
| UC08-VOL-ABS-022-A-N2 | 2 | UC-08 | I would like you to set volume to twenty two percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":22}` |
| UC08-VOL-ABS-023-A | 1 | UC-08 | set volume to twenty three percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":23}` |
| UC08-VOL-ABS-023-A-N1 | 2 | UC-08 | could you set volume to twenty three percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":23}` |
| UC08-VOL-ABS-023-A-N2 | 2 | UC-08 | I would like you to set volume to twenty three percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":23}` |
| UC08-VOL-ABS-024-A | 1 | UC-08 | set volume to twenty four percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":24}` |
| UC08-VOL-ABS-024-A-N1 | 2 | UC-08 | could you set volume to twenty four percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":24}` |
| UC08-VOL-ABS-024-A-N2 | 2 | UC-08 | I would like you to set volume to twenty four percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":24}` |
| UC08-VOL-ABS-025-A | 1 | UC-08 | set volume to twenty five percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":25}` |
| UC08-VOL-ABS-025-A-N1 | 2 | UC-08 | could you set volume to twenty five percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":25}` |
| UC08-VOL-ABS-025-A-N2 | 2 | UC-08 | I would like you to set volume to twenty five percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":25}` |
| UC08-VOL-ABS-026-A | 1 | UC-08 | set volume to twenty six percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":26}` |
| UC08-VOL-ABS-026-A-N1 | 2 | UC-08 | could you set volume to twenty six percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":26}` |
| UC08-VOL-ABS-026-A-N2 | 2 | UC-08 | I would like you to set volume to twenty six percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":26}` |
| UC08-VOL-ABS-027-A | 1 | UC-08 | set volume to twenty seven percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":27}` |
| UC08-VOL-ABS-027-A-N1 | 2 | UC-08 | could you set volume to twenty seven percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":27}` |
| UC08-VOL-ABS-027-A-N2 | 2 | UC-08 | I would like you to set volume to twenty seven percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":27}` |
| UC08-VOL-ABS-028-A | 1 | UC-08 | set volume to twenty eight percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":28}` |
| UC08-VOL-ABS-028-A-N1 | 2 | UC-08 | could you set volume to twenty eight percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":28}` |
| UC08-VOL-ABS-028-A-N2 | 2 | UC-08 | I would like you to set volume to twenty eight percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":28}` |
| UC08-VOL-ABS-029-A | 1 | UC-08 | set volume to twenty nine percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":29}` |
| UC08-VOL-ABS-029-A-N1 | 2 | UC-08 | could you set volume to twenty nine percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":29}` |
| UC08-VOL-ABS-029-A-N2 | 2 | UC-08 | I would like you to set volume to twenty nine percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":29}` |
| UC08-VOL-ABS-030-A | 1 | UC-08 | set volume to thirty percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":30}` |
| UC08-VOL-ABS-030-A-N1 | 2 | UC-08 | could you set volume to thirty percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":30}` |
| UC08-VOL-ABS-030-A-N2 | 2 | UC-08 | I would like you to set volume to thirty percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":30}` |
| UC08-VOL-ABS-031-A | 1 | UC-08 | set volume to thirty one percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":31}` |
| UC08-VOL-ABS-031-A-N1 | 2 | UC-08 | could you set volume to thirty one percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":31}` |
| UC08-VOL-ABS-031-A-N2 | 2 | UC-08 | I would like you to set volume to thirty one percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":31}` |
| UC08-VOL-ABS-032-A | 1 | UC-08 | set volume to thirty two percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":32}` |
| UC08-VOL-ABS-032-A-N1 | 2 | UC-08 | could you set volume to thirty two percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":32}` |
| UC08-VOL-ABS-032-A-N2 | 2 | UC-08 | I would like you to set volume to thirty two percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":32}` |
| UC08-VOL-ABS-033-A | 1 | UC-08 | set volume to thirty three percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":33}` |
| UC08-VOL-ABS-033-A-N1 | 2 | UC-08 | could you set volume to thirty three percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":33}` |
| UC08-VOL-ABS-033-A-N2 | 2 | UC-08 | I would like you to set volume to thirty three percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":33}` |
| UC08-VOL-ABS-034-A | 1 | UC-08 | set volume to thirty four percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":34}` |
| UC08-VOL-ABS-034-A-N1 | 2 | UC-08 | could you set volume to thirty four percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":34}` |
| UC08-VOL-ABS-034-A-N2 | 2 | UC-08 | I would like you to set volume to thirty four percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":34}` |
| UC08-VOL-ABS-035-A | 1 | UC-08 | set volume to thirty five percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":35}` |
| UC08-VOL-ABS-035-A-N1 | 2 | UC-08 | could you set volume to thirty five percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":35}` |
| UC08-VOL-ABS-035-A-N2 | 2 | UC-08 | I would like you to set volume to thirty five percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":35}` |
| UC08-VOL-ABS-036-A | 1 | UC-08 | set volume to thirty six percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":36}` |
| UC08-VOL-ABS-036-A-N1 | 2 | UC-08 | could you set volume to thirty six percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":36}` |
| UC08-VOL-ABS-036-A-N2 | 2 | UC-08 | I would like you to set volume to thirty six percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":36}` |
| UC08-VOL-ABS-037-A | 1 | UC-08 | set volume to thirty seven percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":37}` |
| UC08-VOL-ABS-037-A-N1 | 2 | UC-08 | could you set volume to thirty seven percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":37}` |
| UC08-VOL-ABS-037-A-N2 | 2 | UC-08 | I would like you to set volume to thirty seven percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":37}` |
| UC08-VOL-ABS-038-A | 1 | UC-08 | set volume to thirty eight percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":38}` |
| UC08-VOL-ABS-038-A-N1 | 2 | UC-08 | could you set volume to thirty eight percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":38}` |
| UC08-VOL-ABS-038-A-N2 | 2 | UC-08 | I would like you to set volume to thirty eight percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":38}` |
| UC08-VOL-ABS-039-A | 1 | UC-08 | set volume to thirty nine percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":39}` |
| UC08-VOL-ABS-039-A-N1 | 2 | UC-08 | could you set volume to thirty nine percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":39}` |
| UC08-VOL-ABS-039-A-N2 | 2 | UC-08 | I would like you to set volume to thirty nine percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":39}` |
| UC08-VOL-ABS-040-A | 1 | UC-08 | set volume to forty percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":40}` |
| UC08-VOL-ABS-040-A-N1 | 2 | UC-08 | could you set volume to forty percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":40}` |
| UC08-VOL-ABS-040-A-N2 | 2 | UC-08 | I would like you to set volume to forty percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":40}` |
| UC08-VOL-ABS-041-A | 1 | UC-08 | set volume to forty one percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":41}` |
| UC08-VOL-ABS-041-A-N1 | 2 | UC-08 | could you set volume to forty one percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":41}` |
| UC08-VOL-ABS-041-A-N2 | 2 | UC-08 | I would like you to set volume to forty one percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":41}` |
| UC08-VOL-ABS-042-A | 1 | UC-08 | set volume to forty two percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":42}` |
| UC08-VOL-ABS-042-A-N1 | 2 | UC-08 | could you set volume to forty two percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":42}` |
| UC08-VOL-ABS-042-A-N2 | 2 | UC-08 | I would like you to set volume to forty two percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":42}` |
| UC08-VOL-ABS-043-A | 1 | UC-08 | set volume to forty three percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":43}` |
| UC08-VOL-ABS-043-A-N1 | 2 | UC-08 | could you set volume to forty three percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":43}` |
| UC08-VOL-ABS-043-A-N2 | 2 | UC-08 | I would like you to set volume to forty three percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":43}` |
| UC08-VOL-ABS-044-A | 1 | UC-08 | set volume to forty four percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":44}` |
| UC08-VOL-ABS-044-A-N1 | 2 | UC-08 | could you set volume to forty four percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":44}` |
| UC08-VOL-ABS-044-A-N2 | 2 | UC-08 | I would like you to set volume to forty four percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":44}` |
| UC08-VOL-ABS-045-A | 1 | UC-08 | set volume to forty five percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":45}` |
| UC08-VOL-ABS-045-A-N1 | 2 | UC-08 | could you set volume to forty five percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":45}` |
| UC08-VOL-ABS-045-A-N2 | 2 | UC-08 | I would like you to set volume to forty five percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":45}` |
| UC08-VOL-ABS-046-A | 1 | UC-08 | set volume to forty six percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":46}` |
| UC08-VOL-ABS-046-A-N1 | 2 | UC-08 | could you set volume to forty six percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":46}` |
| UC08-VOL-ABS-046-A-N2 | 2 | UC-08 | I would like you to set volume to forty six percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":46}` |
| UC08-VOL-ABS-047-A | 1 | UC-08 | set volume to forty seven percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":47}` |
| UC08-VOL-ABS-047-A-N1 | 2 | UC-08 | could you set volume to forty seven percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":47}` |
| UC08-VOL-ABS-047-A-N2 | 2 | UC-08 | I would like you to set volume to forty seven percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":47}` |
| UC08-VOL-ABS-048-A | 1 | UC-08 | set volume to forty eight percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":48}` |
| UC08-VOL-ABS-048-A-N1 | 2 | UC-08 | could you set volume to forty eight percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":48}` |
| UC08-VOL-ABS-048-A-N2 | 2 | UC-08 | I would like you to set volume to forty eight percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":48}` |
| UC08-VOL-ABS-049-A | 1 | UC-08 | set volume to forty nine percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":49}` |
| UC08-VOL-ABS-049-A-N1 | 2 | UC-08 | could you set volume to forty nine percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":49}` |
| UC08-VOL-ABS-049-A-N2 | 2 | UC-08 | I would like you to set volume to forty nine percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":49}` |
| UC08-VOL-ABS-050-A | 1 | UC-08 | set volume to fifty percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":50}` |
| UC08-VOL-ABS-050-A-N1 | 2 | UC-08 | could you set volume to fifty percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":50}` |
| UC08-VOL-ABS-050-A-N2 | 2 | UC-08 | I would like you to set volume to fifty percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":50}` |
| UC08-VOL-ABS-051-A | 1 | UC-08 | set volume to fifty one percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":51}` |
| UC08-VOL-ABS-051-A-N1 | 2 | UC-08 | could you set volume to fifty one percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":51}` |
| UC08-VOL-ABS-051-A-N2 | 2 | UC-08 | I would like you to set volume to fifty one percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":51}` |
| UC08-VOL-ABS-052-A | 1 | UC-08 | set volume to fifty two percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":52}` |
| UC08-VOL-ABS-052-A-N1 | 2 | UC-08 | could you set volume to fifty two percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":52}` |
| UC08-VOL-ABS-052-A-N2 | 2 | UC-08 | I would like you to set volume to fifty two percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":52}` |
| UC08-VOL-ABS-053-A | 1 | UC-08 | set volume to fifty three percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":53}` |
| UC08-VOL-ABS-053-A-N1 | 2 | UC-08 | could you set volume to fifty three percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":53}` |
| UC08-VOL-ABS-053-A-N2 | 2 | UC-08 | I would like you to set volume to fifty three percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":53}` |
| UC08-VOL-ABS-054-A | 1 | UC-08 | set volume to fifty four percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":54}` |
| UC08-VOL-ABS-054-A-N1 | 2 | UC-08 | could you set volume to fifty four percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":54}` |
| UC08-VOL-ABS-054-A-N2 | 2 | UC-08 | I would like you to set volume to fifty four percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":54}` |
| UC08-VOL-ABS-055-A | 1 | UC-08 | set volume to fifty five percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":55}` |
| UC08-VOL-ABS-055-A-N1 | 2 | UC-08 | could you set volume to fifty five percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":55}` |
| UC08-VOL-ABS-055-A-N2 | 2 | UC-08 | I would like you to set volume to fifty five percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":55}` |
| UC08-VOL-ABS-056-A | 1 | UC-08 | set volume to fifty six percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":56}` |
| UC08-VOL-ABS-056-A-N1 | 2 | UC-08 | could you set volume to fifty six percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":56}` |
| UC08-VOL-ABS-056-A-N2 | 2 | UC-08 | I would like you to set volume to fifty six percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":56}` |
| UC08-VOL-ABS-057-A | 1 | UC-08 | set volume to fifty seven percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":57}` |
| UC08-VOL-ABS-057-A-N1 | 2 | UC-08 | could you set volume to fifty seven percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":57}` |
| UC08-VOL-ABS-057-A-N2 | 2 | UC-08 | I would like you to set volume to fifty seven percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":57}` |
| UC08-VOL-ABS-058-A | 1 | UC-08 | set volume to fifty eight percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":58}` |
| UC08-VOL-ABS-058-A-N1 | 2 | UC-08 | could you set volume to fifty eight percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":58}` |
| UC08-VOL-ABS-058-A-N2 | 2 | UC-08 | I would like you to set volume to fifty eight percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":58}` |
| UC08-VOL-ABS-059-A | 1 | UC-08 | set volume to fifty nine percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":59}` |
| UC08-VOL-ABS-059-A-N1 | 2 | UC-08 | could you set volume to fifty nine percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":59}` |
| UC08-VOL-ABS-059-A-N2 | 2 | UC-08 | I would like you to set volume to fifty nine percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":59}` |
| UC08-VOL-ABS-060-A | 1 | UC-08 | set volume to sixty percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":60}` |
| UC08-VOL-ABS-060-A-N1 | 2 | UC-08 | could you set volume to sixty percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":60}` |
| UC08-VOL-ABS-060-A-N2 | 2 | UC-08 | I would like you to set volume to sixty percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":60}` |
| UC08-VOL-ABS-061-A | 1 | UC-08 | set volume to sixty one percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":61}` |
| UC08-VOL-ABS-061-A-N1 | 2 | UC-08 | could you set volume to sixty one percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":61}` |
| UC08-VOL-ABS-061-A-N2 | 2 | UC-08 | I would like you to set volume to sixty one percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":61}` |
| UC08-VOL-ABS-062-A | 1 | UC-08 | set volume to sixty two percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":62}` |
| UC08-VOL-ABS-062-A-N1 | 2 | UC-08 | could you set volume to sixty two percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":62}` |
| UC08-VOL-ABS-062-A-N2 | 2 | UC-08 | I would like you to set volume to sixty two percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":62}` |
| UC08-VOL-ABS-063-A | 1 | UC-08 | set volume to sixty three percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":63}` |
| UC08-VOL-ABS-063-A-N1 | 2 | UC-08 | could you set volume to sixty three percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":63}` |
| UC08-VOL-ABS-063-A-N2 | 2 | UC-08 | I would like you to set volume to sixty three percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":63}` |
| UC08-VOL-ABS-064-A | 1 | UC-08 | set volume to sixty four percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":64}` |
| UC08-VOL-ABS-064-A-N1 | 2 | UC-08 | could you set volume to sixty four percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":64}` |
| UC08-VOL-ABS-064-A-N2 | 2 | UC-08 | I would like you to set volume to sixty four percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":64}` |
| UC08-VOL-ABS-065-A | 1 | UC-08 | set volume to sixty five percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":65}` |
| UC08-VOL-ABS-065-A-N1 | 2 | UC-08 | could you set volume to sixty five percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":65}` |
| UC08-VOL-ABS-065-A-N2 | 2 | UC-08 | I would like you to set volume to sixty five percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":65}` |
| UC08-VOL-ABS-066-A | 1 | UC-08 | set volume to sixty six percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":66}` |
| UC08-VOL-ABS-066-A-N1 | 2 | UC-08 | could you set volume to sixty six percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":66}` |
| UC08-VOL-ABS-066-A-N2 | 2 | UC-08 | I would like you to set volume to sixty six percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":66}` |
| UC08-VOL-ABS-067-A | 1 | UC-08 | set volume to sixty seven percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":67}` |
| UC08-VOL-ABS-067-A-N1 | 2 | UC-08 | could you set volume to sixty seven percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":67}` |
| UC08-VOL-ABS-067-A-N2 | 2 | UC-08 | I would like you to set volume to sixty seven percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":67}` |
| UC08-VOL-ABS-068-A | 1 | UC-08 | set volume to sixty eight percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":68}` |
| UC08-VOL-ABS-068-A-N1 | 2 | UC-08 | could you set volume to sixty eight percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":68}` |
| UC08-VOL-ABS-068-A-N2 | 2 | UC-08 | I would like you to set volume to sixty eight percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":68}` |
| UC08-VOL-ABS-069-A | 1 | UC-08 | set volume to sixty nine percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":69}` |
| UC08-VOL-ABS-069-A-N1 | 2 | UC-08 | could you set volume to sixty nine percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":69}` |
| UC08-VOL-ABS-069-A-N2 | 2 | UC-08 | I would like you to set volume to sixty nine percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":69}` |
| UC08-VOL-ABS-070-A | 1 | UC-08 | set volume to seventy percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":70}` |
| UC08-VOL-ABS-070-A-N1 | 2 | UC-08 | could you set volume to seventy percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":70}` |
| UC08-VOL-ABS-070-A-N2 | 2 | UC-08 | I would like you to set volume to seventy percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":70}` |
| UC08-VOL-ABS-071-A | 1 | UC-08 | set volume to seventy one percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":71}` |
| UC08-VOL-ABS-071-A-N1 | 2 | UC-08 | could you set volume to seventy one percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":71}` |
| UC08-VOL-ABS-071-A-N2 | 2 | UC-08 | I would like you to set volume to seventy one percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":71}` |
| UC08-VOL-ABS-072-A | 1 | UC-08 | set volume to seventy two percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":72}` |
| UC08-VOL-ABS-072-A-N1 | 2 | UC-08 | could you set volume to seventy two percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":72}` |
| UC08-VOL-ABS-072-A-N2 | 2 | UC-08 | I would like you to set volume to seventy two percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":72}` |
| UC08-VOL-ABS-073-A | 1 | UC-08 | set volume to seventy three percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":73}` |
| UC08-VOL-ABS-073-A-N1 | 2 | UC-08 | could you set volume to seventy three percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":73}` |
| UC08-VOL-ABS-073-A-N2 | 2 | UC-08 | I would like you to set volume to seventy three percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":73}` |
| UC08-VOL-ABS-074-A | 1 | UC-08 | set volume to seventy four percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":74}` |
| UC08-VOL-ABS-074-A-N1 | 2 | UC-08 | could you set volume to seventy four percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":74}` |
| UC08-VOL-ABS-074-A-N2 | 2 | UC-08 | I would like you to set volume to seventy four percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":74}` |
| UC08-VOL-ABS-075-A | 1 | UC-08 | set volume to seventy five percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":75}` |
| UC08-VOL-ABS-075-A-N1 | 2 | UC-08 | could you set volume to seventy five percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":75}` |
| UC08-VOL-ABS-075-A-N2 | 2 | UC-08 | I would like you to set volume to seventy five percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":75}` |
| UC08-VOL-ABS-076-A | 1 | UC-08 | set volume to seventy six percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":76}` |
| UC08-VOL-ABS-076-A-N1 | 2 | UC-08 | could you set volume to seventy six percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":76}` |
| UC08-VOL-ABS-076-A-N2 | 2 | UC-08 | I would like you to set volume to seventy six percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":76}` |
| UC08-VOL-ABS-077-A | 1 | UC-08 | set volume to seventy seven percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":77}` |
| UC08-VOL-ABS-077-A-N1 | 2 | UC-08 | could you set volume to seventy seven percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":77}` |
| UC08-VOL-ABS-077-A-N2 | 2 | UC-08 | I would like you to set volume to seventy seven percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":77}` |
| UC08-VOL-ABS-078-A | 1 | UC-08 | set volume to seventy eight percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":78}` |
| UC08-VOL-ABS-078-A-N1 | 2 | UC-08 | could you set volume to seventy eight percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":78}` |
| UC08-VOL-ABS-078-A-N2 | 2 | UC-08 | I would like you to set volume to seventy eight percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":78}` |
| UC08-VOL-ABS-079-A | 1 | UC-08 | set volume to seventy nine percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":79}` |
| UC08-VOL-ABS-079-A-N1 | 2 | UC-08 | could you set volume to seventy nine percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":79}` |
| UC08-VOL-ABS-079-A-N2 | 2 | UC-08 | I would like you to set volume to seventy nine percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":79}` |
| UC08-VOL-ABS-080-A | 1 | UC-08 | set volume to eighty percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":80}` |
| UC08-VOL-ABS-080-A-N1 | 2 | UC-08 | could you set volume to eighty percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":80}` |
| UC08-VOL-ABS-080-A-N2 | 2 | UC-08 | I would like you to set volume to eighty percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":80}` |
| UC08-VOL-ABS-081-A | 1 | UC-08 | set volume to eighty one percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":81}` |
| UC08-VOL-ABS-081-A-N1 | 2 | UC-08 | could you set volume to eighty one percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":81}` |
| UC08-VOL-ABS-081-A-N2 | 2 | UC-08 | I would like you to set volume to eighty one percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":81}` |
| UC08-VOL-ABS-082-A | 1 | UC-08 | set volume to eighty two percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":82}` |
| UC08-VOL-ABS-082-A-N1 | 2 | UC-08 | could you set volume to eighty two percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":82}` |
| UC08-VOL-ABS-082-A-N2 | 2 | UC-08 | I would like you to set volume to eighty two percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":82}` |
| UC08-VOL-ABS-083-A | 1 | UC-08 | set volume to eighty three percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":83}` |
| UC08-VOL-ABS-083-A-N1 | 2 | UC-08 | could you set volume to eighty three percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":83}` |
| UC08-VOL-ABS-083-A-N2 | 2 | UC-08 | I would like you to set volume to eighty three percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":83}` |
| UC08-VOL-ABS-084-A | 1 | UC-08 | set volume to eighty four percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":84}` |
| UC08-VOL-ABS-084-A-N1 | 2 | UC-08 | could you set volume to eighty four percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":84}` |
| UC08-VOL-ABS-084-A-N2 | 2 | UC-08 | I would like you to set volume to eighty four percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":84}` |
| UC08-VOL-ABS-085-A | 1 | UC-08 | set volume to eighty five percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":85}` |
| UC08-VOL-ABS-085-A-N1 | 2 | UC-08 | could you set volume to eighty five percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":85}` |
| UC08-VOL-ABS-085-A-N2 | 2 | UC-08 | I would like you to set volume to eighty five percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":85}` |
| UC08-VOL-ABS-086-A | 1 | UC-08 | set volume to eighty six percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":86}` |
| UC08-VOL-ABS-086-A-N1 | 2 | UC-08 | could you set volume to eighty six percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":86}` |
| UC08-VOL-ABS-086-A-N2 | 2 | UC-08 | I would like you to set volume to eighty six percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":86}` |
| UC08-VOL-ABS-087-A | 1 | UC-08 | set volume to eighty seven percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":87}` |
| UC08-VOL-ABS-087-A-N1 | 2 | UC-08 | could you set volume to eighty seven percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":87}` |
| UC08-VOL-ABS-087-A-N2 | 2 | UC-08 | I would like you to set volume to eighty seven percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":87}` |
| UC08-VOL-ABS-088-A | 1 | UC-08 | set volume to eighty eight percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":88}` |
| UC08-VOL-ABS-088-A-N1 | 2 | UC-08 | could you set volume to eighty eight percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":88}` |
| UC08-VOL-ABS-088-A-N2 | 2 | UC-08 | I would like you to set volume to eighty eight percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":88}` |
| UC08-VOL-ABS-089-A | 1 | UC-08 | set volume to eighty nine percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":89}` |
| UC08-VOL-ABS-089-A-N1 | 2 | UC-08 | could you set volume to eighty nine percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":89}` |
| UC08-VOL-ABS-089-A-N2 | 2 | UC-08 | I would like you to set volume to eighty nine percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":89}` |
| UC08-VOL-ABS-090-A | 1 | UC-08 | set volume to ninety percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":90}` |
| UC08-VOL-ABS-090-A-N1 | 2 | UC-08 | could you set volume to ninety percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":90}` |
| UC08-VOL-ABS-090-A-N2 | 2 | UC-08 | I would like you to set volume to ninety percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":90}` |
| UC08-VOL-ABS-091-A | 1 | UC-08 | set volume to ninety one percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":91}` |
| UC08-VOL-ABS-091-A-N1 | 2 | UC-08 | could you set volume to ninety one percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":91}` |
| UC08-VOL-ABS-091-A-N2 | 2 | UC-08 | I would like you to set volume to ninety one percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":91}` |
| UC08-VOL-ABS-092-A | 1 | UC-08 | set volume to ninety two percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":92}` |
| UC08-VOL-ABS-092-A-N1 | 2 | UC-08 | could you set volume to ninety two percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":92}` |
| UC08-VOL-ABS-092-A-N2 | 2 | UC-08 | I would like you to set volume to ninety two percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":92}` |
| UC08-VOL-ABS-093-A | 1 | UC-08 | set volume to ninety three percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":93}` |
| UC08-VOL-ABS-093-A-N1 | 2 | UC-08 | could you set volume to ninety three percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":93}` |
| UC08-VOL-ABS-093-A-N2 | 2 | UC-08 | I would like you to set volume to ninety three percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":93}` |
| UC08-VOL-ABS-094-A | 1 | UC-08 | set volume to ninety four percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":94}` |
| UC08-VOL-ABS-094-A-N1 | 2 | UC-08 | could you set volume to ninety four percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":94}` |
| UC08-VOL-ABS-094-A-N2 | 2 | UC-08 | I would like you to set volume to ninety four percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":94}` |
| UC08-VOL-ABS-095-A | 1 | UC-08 | set volume to ninety five percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":95}` |
| UC08-VOL-ABS-095-A-N1 | 2 | UC-08 | could you set volume to ninety five percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":95}` |
| UC08-VOL-ABS-095-A-N2 | 2 | UC-08 | I would like you to set volume to ninety five percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":95}` |
| UC08-VOL-ABS-096-A | 1 | UC-08 | set volume to ninety six percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":96}` |
| UC08-VOL-ABS-096-A-N1 | 2 | UC-08 | could you set volume to ninety six percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":96}` |
| UC08-VOL-ABS-096-A-N2 | 2 | UC-08 | I would like you to set volume to ninety six percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":96}` |
| UC08-VOL-ABS-097-A | 1 | UC-08 | set volume to ninety seven percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":97}` |
| UC08-VOL-ABS-097-A-N1 | 2 | UC-08 | could you set volume to ninety seven percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":97}` |
| UC08-VOL-ABS-097-A-N2 | 2 | UC-08 | I would like you to set volume to ninety seven percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":97}` |
| UC08-VOL-ABS-098-A | 1 | UC-08 | set volume to ninety eight percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":98}` |
| UC08-VOL-ABS-098-A-N1 | 2 | UC-08 | could you set volume to ninety eight percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":98}` |
| UC08-VOL-ABS-098-A-N2 | 2 | UC-08 | I would like you to set volume to ninety eight percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":98}` |
| UC08-VOL-ABS-099-A | 1 | UC-08 | set volume to ninety nine percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":99}` |
| UC08-VOL-ABS-099-A-N1 | 2 | UC-08 | could you set volume to ninety nine percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":99}` |
| UC08-VOL-ABS-099-A-N2 | 2 | UC-08 | I would like you to set volume to ninety nine percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":99}` |
| UC08-VOL-ABS-100-A | 1 | UC-08 | set volume to one hundred percent | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":100}` |
| UC08-VOL-ABS-100-A-N1 | 2 | UC-08 | could you set volume to one hundred percent, please | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":100}` |
| UC08-VOL-ABS-100-A-N2 | 2 | UC-08 | I would like you to set volume to one hundred percent for me | `{"entity_id":"media.main","intent":"VOLUME_ABSOLUTE","value":100}` |
| UC08-VOL-DELTA-P05 | 1 | UC-08 | increase volume by five percent | `{"delta":5,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DELTA-P05-N1 | 2 | UC-08 | could you increase volume by five percent, please | `{"delta":5,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DELTA-P05-N2 | 2 | UC-08 | I would like you to increase volume by five percent for me | `{"delta":5,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DELTA-P10 | 1 | UC-08 | increase volume by ten percent | `{"delta":10,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DELTA-P10-N1 | 2 | UC-08 | could you increase volume by ten percent, please | `{"delta":10,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DELTA-P10-N2 | 2 | UC-08 | I would like you to increase volume by ten percent for me | `{"delta":10,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DELTA-M05 | 1 | UC-08 | decrease volume by five percent | `{"delta":-5,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DELTA-M05-N1 | 2 | UC-08 | could you decrease volume by five percent, please | `{"delta":-5,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DELTA-M05-N2 | 2 | UC-08 | I would like you to decrease volume by five percent for me | `{"delta":-5,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DELTA-M10 | 1 | UC-08 | decrease volume by ten percent | `{"delta":-10,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DELTA-M10-N1 | 2 | UC-08 | could you decrease volume by ten percent, please | `{"delta":-10,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DELTA-M10-N2 | 2 | UC-08 | I would like you to decrease volume by ten percent for me | `{"delta":-10,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-UP | 1 | UC-08 | volume up | `{"delta":5,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-UP-N1 | 2 | UC-08 | could you turn the volume up, please | `{"delta":5,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-UP-N2 | 2 | UC-08 | I would like you to turn the volume up for me | `{"delta":5,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DOWN | 1 | UC-08 | volume down | `{"delta":-5,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DOWN-N1 | 2 | UC-08 | could you turn the volume down, please | `{"delta":-5,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC08-VOL-DOWN-N2 | 2 | UC-08 | I would like you to turn the volume down for me | `{"delta":-5,"entity_id":"media.main","intent":"VOLUME_DELTA"}` |
| UC09-REM-WATER-A | 1 | UC-09 | remind me to drink water in ten minutes | `{"content_id":"reminder_content.water","delay_seconds":600,"intent":"REMINDER_SET","reminder_id":"reminder.water"}` |
| UC09-REM-WATER-A-N1 | 2 | UC-09 | could you remind me to drink water in ten minutes, please | `{"content_id":"reminder_content.water","delay_seconds":600,"intent":"REMINDER_SET","reminder_id":"reminder.water"}` |
| UC09-REM-WATER-A-N2 | 2 | UC-09 | I would like you to remind me to drink water in ten minutes for me | `{"content_id":"reminder_content.water","delay_seconds":600,"intent":"REMINDER_SET","reminder_id":"reminder.water"}` |
| UC09-REM-WATER-B | 1 | UC-09 | set a reminder to drink water in ten minutes | `{"content_id":"reminder_content.water","delay_seconds":600,"intent":"REMINDER_SET","reminder_id":"reminder.water"}` |
| UC09-REM-WATER-B-N1 | 2 | UC-09 | could you set a reminder to drink water in ten minutes, please | `{"content_id":"reminder_content.water","delay_seconds":600,"intent":"REMINDER_SET","reminder_id":"reminder.water"}` |
| UC09-REM-WATER-B-N2 | 2 | UC-09 | I would like you to set a reminder to drink water in ten minutes for me | `{"content_id":"reminder_content.water","delay_seconds":600,"intent":"REMINDER_SET","reminder_id":"reminder.water"}` |
| UC09-REM-CANCEL-WATER | 1 | UC-09 | cancel my water reminder | `{"intent":"REMINDER_CANCEL","reminder_id":"reminder.water"}` |
| UC09-REM-CANCEL-WATER-N1 | 2 | UC-09 | could you cancel my water reminder, please | `{"intent":"REMINDER_CANCEL","reminder_id":"reminder.water"}` |
| UC09-REM-CANCEL-WATER-N2 | 2 | UC-09 | I would like you to cancel my water reminder for me | `{"intent":"REMINDER_CANCEL","reminder_id":"reminder.water"}` |
| UC09-REM-OVEN-A | 1 | UC-09 | remind me to check the oven in five minutes | `{"content_id":"reminder_content.oven","delay_seconds":300,"intent":"REMINDER_SET","reminder_id":"reminder.oven"}` |
| UC09-REM-OVEN-A-N1 | 2 | UC-09 | could you remind me to check the oven in five minutes, please | `{"content_id":"reminder_content.oven","delay_seconds":300,"intent":"REMINDER_SET","reminder_id":"reminder.oven"}` |
| UC09-REM-OVEN-A-N2 | 2 | UC-09 | I would like you to remind me to check the oven in five minutes for me | `{"content_id":"reminder_content.oven","delay_seconds":300,"intent":"REMINDER_SET","reminder_id":"reminder.oven"}` |
| UC09-REM-OVEN-B | 1 | UC-09 | set a reminder to check the oven in five minutes | `{"content_id":"reminder_content.oven","delay_seconds":300,"intent":"REMINDER_SET","reminder_id":"reminder.oven"}` |
| UC09-REM-OVEN-B-N1 | 2 | UC-09 | could you set a reminder to check the oven in five minutes, please | `{"content_id":"reminder_content.oven","delay_seconds":300,"intent":"REMINDER_SET","reminder_id":"reminder.oven"}` |
| UC09-REM-OVEN-B-N2 | 2 | UC-09 | I would like you to set a reminder to check the oven in five minutes for me | `{"content_id":"reminder_content.oven","delay_seconds":300,"intent":"REMINDER_SET","reminder_id":"reminder.oven"}` |
| UC09-REM-CANCEL-OVEN | 1 | UC-09 | cancel my oven reminder | `{"intent":"REMINDER_CANCEL","reminder_id":"reminder.oven"}` |
| UC09-REM-CANCEL-OVEN-N1 | 2 | UC-09 | could you cancel my oven reminder, please | `{"intent":"REMINDER_CANCEL","reminder_id":"reminder.oven"}` |
| UC09-REM-CANCEL-OVEN-N2 | 2 | UC-09 | I would like you to cancel my oven reminder for me | `{"intent":"REMINDER_CANCEL","reminder_id":"reminder.oven"}` |
| UC09-REM-STRETCH-A | 1 | UC-09 | remind me to stretch in thirty minutes | `{"content_id":"reminder_content.stretch","delay_seconds":1800,"intent":"REMINDER_SET","reminder_id":"reminder.stretch"}` |
| UC09-REM-STRETCH-A-N1 | 2 | UC-09 | could you remind me to stretch in thirty minutes, please | `{"content_id":"reminder_content.stretch","delay_seconds":1800,"intent":"REMINDER_SET","reminder_id":"reminder.stretch"}` |
| UC09-REM-STRETCH-A-N2 | 2 | UC-09 | I would like you to remind me to stretch in thirty minutes for me | `{"content_id":"reminder_content.stretch","delay_seconds":1800,"intent":"REMINDER_SET","reminder_id":"reminder.stretch"}` |
| UC09-REM-STRETCH-B | 1 | UC-09 | set a reminder to stretch in thirty minutes | `{"content_id":"reminder_content.stretch","delay_seconds":1800,"intent":"REMINDER_SET","reminder_id":"reminder.stretch"}` |
| UC09-REM-STRETCH-B-N1 | 2 | UC-09 | could you set a reminder to stretch in thirty minutes, please | `{"content_id":"reminder_content.stretch","delay_seconds":1800,"intent":"REMINDER_SET","reminder_id":"reminder.stretch"}` |
| UC09-REM-STRETCH-B-N2 | 2 | UC-09 | I would like you to set a reminder to stretch in thirty minutes for me | `{"content_id":"reminder_content.stretch","delay_seconds":1800,"intent":"REMINDER_SET","reminder_id":"reminder.stretch"}` |
| UC09-REM-CANCEL-STRETCH | 1 | UC-09 | cancel my stretch reminder | `{"intent":"REMINDER_CANCEL","reminder_id":"reminder.stretch"}` |
| UC09-REM-CANCEL-STRETCH-N1 | 2 | UC-09 | could you cancel my stretch reminder, please | `{"intent":"REMINDER_CANCEL","reminder_id":"reminder.stretch"}` |
| UC09-REM-CANCEL-STRETCH-N2 | 2 | UC-09 | I would like you to cancel my stretch reminder for me | `{"intent":"REMINDER_CANCEL","reminder_id":"reminder.stretch"}` |
| UC09-LIST-ADD-MILK-A | 1 | UC-09 | add milk to my shopping list | `{"intent":"LIST_ADD","item_id":"item.milk","list_id":"list.shopping"}` |
| UC09-LIST-ADD-MILK-A-N1 | 2 | UC-09 | could you add milk to my shopping list, please | `{"intent":"LIST_ADD","item_id":"item.milk","list_id":"list.shopping"}` |
| UC09-LIST-ADD-MILK-A-N2 | 2 | UC-09 | I would like you to add milk to my shopping list for me | `{"intent":"LIST_ADD","item_id":"item.milk","list_id":"list.shopping"}` |
| UC09-LIST-ADD-MILK-B | 1 | UC-09 | put milk on my shopping list | `{"intent":"LIST_ADD","item_id":"item.milk","list_id":"list.shopping"}` |
| UC09-LIST-ADD-MILK-B-N1 | 2 | UC-09 | could you put milk on my shopping list, please | `{"intent":"LIST_ADD","item_id":"item.milk","list_id":"list.shopping"}` |
| UC09-LIST-ADD-MILK-B-N2 | 2 | UC-09 | I would like you to put milk on my shopping list for me | `{"intent":"LIST_ADD","item_id":"item.milk","list_id":"list.shopping"}` |
| UC09-LIST-REMOVE-MILK | 1 | UC-09 | remove milk from my shopping list | `{"intent":"LIST_REMOVE","item_id":"item.milk","list_id":"list.shopping"}` |
| UC09-LIST-REMOVE-MILK-N1 | 2 | UC-09 | could you remove milk from my shopping list, please | `{"intent":"LIST_REMOVE","item_id":"item.milk","list_id":"list.shopping"}` |
| UC09-LIST-REMOVE-MILK-N2 | 2 | UC-09 | I would like you to remove milk from my shopping list for me | `{"intent":"LIST_REMOVE","item_id":"item.milk","list_id":"list.shopping"}` |
| UC09-LIST-ADD-BREAD-A | 1 | UC-09 | add bread to my shopping list | `{"intent":"LIST_ADD","item_id":"item.bread","list_id":"list.shopping"}` |
| UC09-LIST-ADD-BREAD-A-N1 | 2 | UC-09 | could you add bread to my shopping list, please | `{"intent":"LIST_ADD","item_id":"item.bread","list_id":"list.shopping"}` |
| UC09-LIST-ADD-BREAD-A-N2 | 2 | UC-09 | I would like you to add bread to my shopping list for me | `{"intent":"LIST_ADD","item_id":"item.bread","list_id":"list.shopping"}` |
| UC09-LIST-ADD-BREAD-B | 1 | UC-09 | put bread on my shopping list | `{"intent":"LIST_ADD","item_id":"item.bread","list_id":"list.shopping"}` |
| UC09-LIST-ADD-BREAD-B-N1 | 2 | UC-09 | could you put bread on my shopping list, please | `{"intent":"LIST_ADD","item_id":"item.bread","list_id":"list.shopping"}` |
| UC09-LIST-ADD-BREAD-B-N2 | 2 | UC-09 | I would like you to put bread on my shopping list for me | `{"intent":"LIST_ADD","item_id":"item.bread","list_id":"list.shopping"}` |
| UC09-LIST-REMOVE-BREAD | 1 | UC-09 | remove bread from my shopping list | `{"intent":"LIST_REMOVE","item_id":"item.bread","list_id":"list.shopping"}` |
| UC09-LIST-REMOVE-BREAD-N1 | 2 | UC-09 | could you remove bread from my shopping list, please | `{"intent":"LIST_REMOVE","item_id":"item.bread","list_id":"list.shopping"}` |
| UC09-LIST-REMOVE-BREAD-N2 | 2 | UC-09 | I would like you to remove bread from my shopping list for me | `{"intent":"LIST_REMOVE","item_id":"item.bread","list_id":"list.shopping"}` |
| UC09-LIST-ADD-EGGS-A | 1 | UC-09 | add eggs to my shopping list | `{"intent":"LIST_ADD","item_id":"item.eggs","list_id":"list.shopping"}` |
| UC09-LIST-ADD-EGGS-A-N1 | 2 | UC-09 | could you add eggs to my shopping list, please | `{"intent":"LIST_ADD","item_id":"item.eggs","list_id":"list.shopping"}` |
| UC09-LIST-ADD-EGGS-A-N2 | 2 | UC-09 | I would like you to add eggs to my shopping list for me | `{"intent":"LIST_ADD","item_id":"item.eggs","list_id":"list.shopping"}` |
| UC09-LIST-ADD-EGGS-B | 1 | UC-09 | put eggs on my shopping list | `{"intent":"LIST_ADD","item_id":"item.eggs","list_id":"list.shopping"}` |
| UC09-LIST-ADD-EGGS-B-N1 | 2 | UC-09 | could you put eggs on my shopping list, please | `{"intent":"LIST_ADD","item_id":"item.eggs","list_id":"list.shopping"}` |
| UC09-LIST-ADD-EGGS-B-N2 | 2 | UC-09 | I would like you to put eggs on my shopping list for me | `{"intent":"LIST_ADD","item_id":"item.eggs","list_id":"list.shopping"}` |
| UC09-LIST-REMOVE-EGGS | 1 | UC-09 | remove eggs from my shopping list | `{"intent":"LIST_REMOVE","item_id":"item.eggs","list_id":"list.shopping"}` |
| UC09-LIST-REMOVE-EGGS-N1 | 2 | UC-09 | could you remove eggs from my shopping list, please | `{"intent":"LIST_REMOVE","item_id":"item.eggs","list_id":"list.shopping"}` |
| UC09-LIST-REMOVE-EGGS-N2 | 2 | UC-09 | I would like you to remove eggs from my shopping list for me | `{"intent":"LIST_REMOVE","item_id":"item.eggs","list_id":"list.shopping"}` |
| UC09-LIST-READ-A | 1 | UC-09 | read my shopping list | `{"intent":"LIST_READ","list_id":"list.shopping"}` |
| UC09-LIST-READ-A-N1 | 2 | UC-09 | could you read my shopping list, please | `{"intent":"LIST_READ","list_id":"list.shopping"}` |
| UC09-LIST-READ-A-N2 | 2 | UC-09 | I would like you to read my shopping list for me | `{"intent":"LIST_READ","list_id":"list.shopping"}` |
| UC09-LIST-READ-B | 1 | UC-09 | what is on my shopping list | `{"intent":"LIST_READ","list_id":"list.shopping"}` |
| UC09-LIST-READ-B-N1 | 2 | UC-09 | can you tell me what is on my shopping list, please | `{"intent":"LIST_READ","list_id":"list.shopping"}` |
| UC09-LIST-READ-B-N2 | 2 | UC-09 | I would like to know what is on my shopping list | `{"intent":"LIST_READ","list_id":"list.shopping"}` |
| UC10-CALL-ALEX-A | 1 | UC-10 | call alex kim | `{"contact_id":"contact.alex_kim","intent":"CALL_START"}` |
| UC10-CALL-ALEX-A-N1 | 2 | UC-10 | could you call alex kim, please | `{"contact_id":"contact.alex_kim","intent":"CALL_START"}` |
| UC10-CALL-ALEX-A-N2 | 2 | UC-10 | I would like you to call alex kim for me | `{"contact_id":"contact.alex_kim","intent":"CALL_START"}` |
| UC10-CALL-ALEX-B | 1 | UC-10 | call alex | `{"contact_id":"contact.alex_kim","intent":"CALL_START"}` |
| UC10-CALL-ALEX-B-N1 | 2 | UC-10 | could you call alex, please | `{"contact_id":"contact.alex_kim","intent":"CALL_START"}` |
| UC10-CALL-ALEX-B-N2 | 2 | UC-10 | I would like you to call alex for me | `{"contact_id":"contact.alex_kim","intent":"CALL_START"}` |
| UC10-MSG-ALEX-WAY-A | 1 | UC-10 | message alex kim i am on my way | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-ALEX-WAY-A-N1 | 2 | UC-10 | could you message alex kim i am on my way, please | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-ALEX-WAY-A-N2 | 2 | UC-10 | I would like you to message alex kim i am on my way for me | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-ALEX-WAY-B | 1 | UC-10 | tell alex kim i am on my way | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-ALEX-WAY-B-N1 | 2 | UC-10 | could you tell alex kim i am on my way, please | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-ALEX-WAY-B-N2 | 2 | UC-10 | I would like you to tell alex kim i am on my way for me | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-ALEX-LATE-A | 1 | UC-10 | message alex kim i will be late | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-ALEX-LATE-A-N1 | 2 | UC-10 | could you message alex kim i will be late, please | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-ALEX-LATE-A-N2 | 2 | UC-10 | I would like you to message alex kim i will be late for me | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-ALEX-LATE-B | 1 | UC-10 | tell alex kim i will be late | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-ALEX-LATE-B-N1 | 2 | UC-10 | could you tell alex kim i will be late, please | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-ALEX-LATE-B-N2 | 2 | UC-10 | I would like you to tell alex kim i will be late for me | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-ALEX-CALL-A | 1 | UC-10 | message alex kim please call me | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-ALEX-CALL-A-N1 | 2 | UC-10 | could you message alex kim please call me, please | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-ALEX-CALL-A-N2 | 2 | UC-10 | I would like you to message alex kim please call me for me | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-ALEX-CALL-B | 1 | UC-10 | tell alex kim please call me | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-ALEX-CALL-B-N1 | 2 | UC-10 | could you tell alex kim please call me, please | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-ALEX-CALL-B-N2 | 2 | UC-10 | I would like you to tell alex kim please call me for me | `{"contact_id":"contact.alex_kim","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-CALL-JAMIE-A | 1 | UC-10 | call jamie cruz | `{"contact_id":"contact.jamie_cruz","intent":"CALL_START"}` |
| UC10-CALL-JAMIE-A-N1 | 2 | UC-10 | could you call jamie cruz, please | `{"contact_id":"contact.jamie_cruz","intent":"CALL_START"}` |
| UC10-CALL-JAMIE-A-N2 | 2 | UC-10 | I would like you to call jamie cruz for me | `{"contact_id":"contact.jamie_cruz","intent":"CALL_START"}` |
| UC10-CALL-JAMIE-B | 1 | UC-10 | call jamie | `{"contact_id":"contact.jamie_cruz","intent":"CALL_START"}` |
| UC10-CALL-JAMIE-B-N1 | 2 | UC-10 | could you call jamie, please | `{"contact_id":"contact.jamie_cruz","intent":"CALL_START"}` |
| UC10-CALL-JAMIE-B-N2 | 2 | UC-10 | I would like you to call jamie for me | `{"contact_id":"contact.jamie_cruz","intent":"CALL_START"}` |
| UC10-MSG-JAMIE-WAY-A | 1 | UC-10 | message jamie cruz i am on my way | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-JAMIE-WAY-A-N1 | 2 | UC-10 | could you message jamie cruz i am on my way, please | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-JAMIE-WAY-A-N2 | 2 | UC-10 | I would like you to message jamie cruz i am on my way for me | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-JAMIE-WAY-B | 1 | UC-10 | tell jamie cruz i am on my way | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-JAMIE-WAY-B-N1 | 2 | UC-10 | could you tell jamie cruz i am on my way, please | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-JAMIE-WAY-B-N2 | 2 | UC-10 | I would like you to tell jamie cruz i am on my way for me | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-JAMIE-LATE-A | 1 | UC-10 | message jamie cruz i will be late | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-JAMIE-LATE-A-N1 | 2 | UC-10 | could you message jamie cruz i will be late, please | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-JAMIE-LATE-A-N2 | 2 | UC-10 | I would like you to message jamie cruz i will be late for me | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-JAMIE-LATE-B | 1 | UC-10 | tell jamie cruz i will be late | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-JAMIE-LATE-B-N1 | 2 | UC-10 | could you tell jamie cruz i will be late, please | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-JAMIE-LATE-B-N2 | 2 | UC-10 | I would like you to tell jamie cruz i will be late for me | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-JAMIE-CALL-A | 1 | UC-10 | message jamie cruz please call me | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-JAMIE-CALL-A-N1 | 2 | UC-10 | could you message jamie cruz please call me, please | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-JAMIE-CALL-A-N2 | 2 | UC-10 | I would like you to message jamie cruz please call me for me | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-JAMIE-CALL-B | 1 | UC-10 | tell jamie cruz please call me | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-JAMIE-CALL-B-N1 | 2 | UC-10 | could you tell jamie cruz please call me, please | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-JAMIE-CALL-B-N2 | 2 | UC-10 | I would like you to tell jamie cruz please call me for me | `{"contact_id":"contact.jamie_cruz","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-CALL-MORGAN-A | 1 | UC-10 | call morgan lee | `{"contact_id":"contact.morgan_lee","intent":"CALL_START"}` |
| UC10-CALL-MORGAN-A-N1 | 2 | UC-10 | could you call morgan lee, please | `{"contact_id":"contact.morgan_lee","intent":"CALL_START"}` |
| UC10-CALL-MORGAN-A-N2 | 2 | UC-10 | I would like you to call morgan lee for me | `{"contact_id":"contact.morgan_lee","intent":"CALL_START"}` |
| UC10-CALL-MORGAN-B | 1 | UC-10 | call morgan | `{"contact_id":"contact.morgan_lee","intent":"CALL_START"}` |
| UC10-CALL-MORGAN-B-N1 | 2 | UC-10 | could you call morgan, please | `{"contact_id":"contact.morgan_lee","intent":"CALL_START"}` |
| UC10-CALL-MORGAN-B-N2 | 2 | UC-10 | I would like you to call morgan for me | `{"contact_id":"contact.morgan_lee","intent":"CALL_START"}` |
| UC10-MSG-MORGAN-WAY-A | 1 | UC-10 | message morgan lee i am on my way | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-MORGAN-WAY-A-N1 | 2 | UC-10 | could you message morgan lee i am on my way, please | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-MORGAN-WAY-A-N2 | 2 | UC-10 | I would like you to message morgan lee i am on my way for me | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-MORGAN-WAY-B | 1 | UC-10 | tell morgan lee i am on my way | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-MORGAN-WAY-B-N1 | 2 | UC-10 | could you tell morgan lee i am on my way, please | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-MORGAN-WAY-B-N2 | 2 | UC-10 | I would like you to tell morgan lee i am on my way for me | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.on_my_way"}` |
| UC10-MSG-MORGAN-LATE-A | 1 | UC-10 | message morgan lee i will be late | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-MORGAN-LATE-A-N1 | 2 | UC-10 | could you message morgan lee i will be late, please | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-MORGAN-LATE-A-N2 | 2 | UC-10 | I would like you to message morgan lee i will be late for me | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-MORGAN-LATE-B | 1 | UC-10 | tell morgan lee i will be late | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-MORGAN-LATE-B-N1 | 2 | UC-10 | could you tell morgan lee i will be late, please | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-MORGAN-LATE-B-N2 | 2 | UC-10 | I would like you to tell morgan lee i will be late for me | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.running_late"}` |
| UC10-MSG-MORGAN-CALL-A | 1 | UC-10 | message morgan lee please call me | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-MORGAN-CALL-A-N1 | 2 | UC-10 | could you message morgan lee please call me, please | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-MORGAN-CALL-A-N2 | 2 | UC-10 | I would like you to message morgan lee please call me for me | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-MORGAN-CALL-B | 1 | UC-10 | tell morgan lee please call me | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-MORGAN-CALL-B-N1 | 2 | UC-10 | could you tell morgan lee please call me, please | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC10-MSG-MORGAN-CALL-B-N2 | 2 | UC-10 | I would like you to tell morgan lee please call me for me | `{"contact_id":"contact.morgan_lee","intent":"MESSAGE_SEND","message_id":"message.call_me"}` |
| UC01-REJ-01 | 1 | UC-01 | do not play river by alpha | `null` |
| UC01-REJ-01-N1 | 2 | UC-01 | hey assistant, do not play river by alpha | `null` |
| UC01-REJ-01-N2 | 2 | UC-01 | do not play river by alpha, please | `null` |
| UC01-REJ-02 | 1 | UC-01 | do not play music | `null` |
| UC01-REJ-02-N1 | 2 | UC-01 | hey assistant, do not play music | `null` |
| UC01-REJ-02-N2 | 2 | UC-01 | do not play music, please | `null` |
| UC01-REJ-03 | 1 | UC-01 | play river by alpha and stop music | `null` |
| UC01-REJ-03-N1 | 2 | UC-01 | hey assistant, play river by alpha and stop music | `null` |
| UC01-REJ-03-N2 | 2 | UC-01 | play river by alpha and stop music, please | `null` |
| UC01-REJ-04 | 1 | UC-01 | play favorites playlist and call alex | `null` |
| UC01-REJ-04-N1 | 2 | UC-01 | hey assistant, play favorites playlist and call alex | `null` |
| UC01-REJ-04-N2 | 2 | UC-01 | play favorites playlist and call alex, please | `null` |
| UC01-REJ-05 | 1 | UC-01 | play river | `null` |
| UC01-REJ-05-N1 | 2 | UC-01 | hey assistant, play river | `null` |
| UC01-REJ-05-N2 | 2 | UC-01 | play river, please | `null` |
| UC01-REJ-06 | 1 | UC-01 | play river by | `null` |
| UC01-REJ-06-N1 | 2 | UC-01 | hey assistant, play river by | `null` |
| UC01-REJ-06-N2 | 2 | UC-01 | play river by, please | `null` |
| UC01-REJ-07 | 1 | UC-01 | play | `null` |
| UC01-REJ-07-N1 | 2 | UC-01 | hey assistant, play | `null` |
| UC01-REJ-07-N2 | 2 | UC-01 | play, please | `null` |
| UC01-REJ-08 | 1 | UC-01 | play playlist | `null` |
| UC01-REJ-08-N1 | 2 | UC-01 | hey assistant, play playlist | `null` |
| UC01-REJ-08-N2 | 2 | UC-01 | play playlist, please | `null` |
| UC01-REJ-09 | 1 | UC-01 | river alpha maybe not | `null` |
| UC01-REJ-09-N1 | 2 | UC-01 | hey assistant, river alpha maybe not | `null` |
| UC01-REJ-09-N2 | 2 | UC-01 | river alpha maybe not, please | `null` |
| UC01-REJ-10 | 1 | UC-01 | play river by alpha beta | `null` |
| UC01-REJ-10-N1 | 2 | UC-01 | hey assistant, play river by alpha beta | `null` |
| UC01-REJ-10-N2 | 2 | UC-01 | play river by alpha beta, please | `null` |
| UC01-REJ-11 | 1 | UC-01 | play seventy three by alpha and beta | `null` |
| UC01-REJ-11-N1 | 2 | UC-01 | hey assistant, play seventy three by alpha and beta | `null` |
| UC01-REJ-11-N2 | 2 | UC-01 | play seventy three by alpha and beta, please | `null` |
| UC01-REJ-12 | 1 | UC-01 | play favorites playlist not workout playlist | `null` |
| UC01-REJ-12-N1 | 2 | UC-01 | hey assistant, play favorites playlist not workout playlist | `null` |
| UC01-REJ-12-N2 | 2 | UC-01 | play favorites playlist not workout playlist, please | `null` |
| UC01-REJ-13 | 1 | UC-01 | play sunrise playlist | `null` |
| UC01-REJ-13-N1 | 2 | UC-01 | hey assistant, play sunrise playlist | `null` |
| UC01-REJ-13-N2 | 2 | UC-01 | play sunrise playlist, please | `null` |
| UC01-REJ-14 | 1 | UC-01 | play favorite playlist | `null` |
| UC01-REJ-14-N1 | 2 | UC-01 | hey assistant, play favorite playlist | `null` |
| UC01-REJ-14-N2 | 2 | UC-01 | play favorite playlist, please | `null` |
| UC01-REJ-15 | 1 | UC-01 | play river by gamma | `null` |
| UC01-REJ-15-N1 | 2 | UC-01 | hey assistant, play river by gamma | `null` |
| UC01-REJ-15-N2 | 2 | UC-01 | play river by gamma, please | `null` |
| UC01-REJ-16 | 1 | UC-01 | play seventy four | `null` |
| UC01-REJ-16-N1 | 2 | UC-01 | hey assistant, play seventy four | `null` |
| UC01-REJ-16-N2 | 2 | UC-01 | play seventy four, please | `null` |
| UC01-REJ-17 | 1 | UC-01 | play rain by alpha | `null` |
| UC01-REJ-17-N1 | 2 | UC-01 | hey assistant, play rain by alpha | `null` |
| UC01-REJ-17-N2 | 2 | UC-01 | play rain by alpha, please | `null` |
| UC01-REJ-18 | 1 | UC-01 | shuffle focus playlist | `null` |
| UC01-REJ-18-N1 | 2 | UC-01 | hey assistant, shuffle focus playlist | `null` |
| UC01-REJ-18-N2 | 2 | UC-01 | shuffle focus playlist, please | `null` |
| UC01-REJ-19 | 1 | UC-01 | play some jazz | `null` |
| UC01-REJ-19-N1 | 2 | UC-01 | hey assistant, play some jazz | `null` |
| UC01-REJ-19-N2 | 2 | UC-01 | play some jazz, please | `null` |
| UC01-REJ-20 | 1 | UC-01 | play the favorites playlist | `null` |
| UC01-REJ-20-N1 | 2 | UC-01 | hey assistant, play the favorites playlist | `null` |
| UC01-REJ-20-N2 | 2 | UC-01 | play the favorites playlist, please | `null` |
| UC02-REJ-01 | 1 | UC-02 | do not tell me the time | `null` |
| UC02-REJ-01-N1 | 2 | UC-02 | hey assistant, do not tell me the time | `null` |
| UC02-REJ-01-N2 | 2 | UC-02 | do not tell me the time, please | `null` |
| UC02-REJ-02 | 1 | UC-02 | do not tell me the date | `null` |
| UC02-REJ-02-N1 | 2 | UC-02 | hey assistant, do not tell me the date | `null` |
| UC02-REJ-02-N2 | 2 | UC-02 | do not tell me the date, please | `null` |
| UC02-REJ-03 | 1 | UC-02 | do not tell me the weather | `null` |
| UC02-REJ-03-N1 | 2 | UC-02 | hey assistant, do not tell me the weather | `null` |
| UC02-REJ-03-N2 | 2 | UC-02 | do not tell me the weather, please | `null` |
| UC02-REJ-04 | 1 | UC-02 | what time is it and play music | `null` |
| UC02-REJ-04-N1 | 2 | UC-02 | hey assistant, what time is it and play music | `null` |
| UC02-REJ-04-N2 | 2 | UC-02 | what time is it and play music, please | `null` |
| UC02-REJ-05 | 1 | UC-02 | what is | `null` |
| UC02-REJ-05-N1 | 2 | UC-02 | hey assistant, what is | `null` |
| UC02-REJ-05-N2 | 2 | UC-02 | what is, please | `null` |
| UC02-REJ-06 | 1 | UC-02 | tell me | `null` |
| UC02-REJ-06-N1 | 2 | UC-02 | hey assistant, tell me | `null` |
| UC02-REJ-06-N2 | 2 | UC-02 | tell me, please | `null` |
| UC02-REJ-07 | 1 | UC-02 | what is the weather and call alex | `null` |
| UC02-REJ-07-N1 | 2 | UC-02 | hey assistant, what is the weather and call alex | `null` |
| UC02-REJ-07-N2 | 2 | UC-02 | what is the weather and call alex, please | `null` |
| UC02-REJ-08 | 1 | UC-02 | what is the date and cancel alarm | `null` |
| UC02-REJ-08-N1 | 2 | UC-02 | hey assistant, what is the date and cancel alarm | `null` |
| UC02-REJ-08-N2 | 2 | UC-02 | what is the date and cancel alarm, please | `null` |
| UC02-REJ-09 | 1 | UC-02 | what time date weather | `null` |
| UC02-REJ-09-N1 | 2 | UC-02 | hey assistant, what time date weather | `null` |
| UC02-REJ-09-N2 | 2 | UC-02 | what time date weather, please | `null` |
| UC02-REJ-10 | 1 | UC-02 | weather not weather | `null` |
| UC02-REJ-10-N1 | 2 | UC-02 | hey assistant, weather not weather | `null` |
| UC02-REJ-10-N2 | 2 | UC-02 | weather not weather, please | `null` |
| UC02-REJ-11 | 1 | UC-02 | what is the weather today tomorrow | `null` |
| UC02-REJ-11-N1 | 2 | UC-02 | hey assistant, what is the weather today tomorrow | `null` |
| UC02-REJ-11-N2 | 2 | UC-02 | what is the weather today tomorrow, please | `null` |
| UC02-REJ-12 | 1 | UC-02 | tell me the time not the date | `null` |
| UC02-REJ-12-N1 | 2 | UC-02 | hey assistant, tell me the time not the date | `null` |
| UC02-REJ-12-N2 | 2 | UC-02 | tell me the time not the date, please | `null` |
| UC02-REJ-13 | 1 | UC-02 | what is the weather tomorrow | `null` |
| UC02-REJ-13-N1 | 2 | UC-02 | hey assistant, what is the weather tomorrow | `null` |
| UC02-REJ-13-N2 | 2 | UC-02 | what is the weather tomorrow, please | `null` |
| UC02-REJ-14 | 1 | UC-02 | what is the weather in cebu | `null` |
| UC02-REJ-14-N1 | 2 | UC-02 | hey assistant, what is the weather in cebu | `null` |
| UC02-REJ-14-N2 | 2 | UC-02 | what is the weather in cebu, please | `null` |
| UC02-REJ-15 | 1 | UC-02 | what is the forecast next week | `null` |
| UC02-REJ-15-N1 | 2 | UC-02 | hey assistant, what is the forecast next week | `null` |
| UC02-REJ-15-N2 | 2 | UC-02 | what is the forecast next week, please | `null` |
| UC02-REJ-16 | 1 | UC-02 | is it raining outside | `null` |
| UC02-REJ-16-N1 | 2 | UC-02 | hey assistant, is it raining outside | `null` |
| UC02-REJ-16-N2 | 2 | UC-02 | is it raining outside, please | `null` |
| UC02-REJ-17 | 1 | UC-02 | what is two plus two | `null` |
| UC02-REJ-17-N1 | 2 | UC-02 | hey assistant, what is two plus two | `null` |
| UC02-REJ-17-N2 | 2 | UC-02 | what is two plus two, please | `null` |
| UC02-REJ-18 | 1 | UC-02 | who wrote this song | `null` |
| UC02-REJ-18-N1 | 2 | UC-02 | hey assistant, who wrote this song | `null` |
| UC02-REJ-18-N2 | 2 | UC-02 | who wrote this song, please | `null` |
| UC02-REJ-19 | 1 | UC-02 | what day of the week is it | `null` |
| UC02-REJ-19-N1 | 2 | UC-02 | hey assistant, what day of the week is it | `null` |
| UC02-REJ-19-N2 | 2 | UC-02 | what day of the week is it, please | `null` |
| UC02-REJ-20 | 1 | UC-02 | tell me a joke | `null` |
| UC02-REJ-20-N1 | 2 | UC-02 | hey assistant, tell me a joke | `null` |
| UC02-REJ-20-N2 | 2 | UC-02 | tell me a joke, please | `null` |
| UC03-REJ-01 | 1 | UC-03 | do not turn on kitchen lamp | `null` |
| UC03-REJ-01-N1 | 2 | UC-03 | hey assistant, do not turn on kitchen lamp | `null` |
| UC03-REJ-01-N2 | 2 | UC-03 | do not turn on kitchen lamp, please | `null` |
| UC03-REJ-02 | 1 | UC-03 | do not turn off bedroom lamp | `null` |
| UC03-REJ-02-N1 | 2 | UC-03 | hey assistant, do not turn off bedroom lamp | `null` |
| UC03-REJ-02-N2 | 2 | UC-03 | do not turn off bedroom lamp, please | `null` |
| UC03-REJ-03 | 1 | UC-03 | turn on kitchen lamp and turn off bedroom lamp | `null` |
| UC03-REJ-03-N1 | 2 | UC-03 | hey assistant, turn on kitchen lamp and turn off bedroom lamp | `null` |
| UC03-REJ-03-N2 | 2 | UC-03 | turn on kitchen lamp and turn off bedroom lamp, please | `null` |
| UC03-REJ-04 | 1 | UC-03 | turn on lamp two and call alex | `null` |
| UC03-REJ-04-N1 | 2 | UC-03 | hey assistant, turn on lamp two and call alex | `null` |
| UC03-REJ-04-N2 | 2 | UC-03 | turn on lamp two and call alex, please | `null` |
| UC03-REJ-05 | 1 | UC-03 | turn on | `null` |
| UC03-REJ-05-N1 | 2 | UC-03 | hey assistant, turn on | `null` |
| UC03-REJ-05-N2 | 2 | UC-03 | turn on, please | `null` |
| UC03-REJ-06 | 1 | UC-03 | turn off | `null` |
| UC03-REJ-06-N1 | 2 | UC-03 | hey assistant, turn off | `null` |
| UC03-REJ-06-N2 | 2 | UC-03 | turn off, please | `null` |
| UC03-REJ-07 | 1 | UC-03 | turn on the lamp | `null` |
| UC03-REJ-07-N1 | 2 | UC-03 | hey assistant, turn on the lamp | `null` |
| UC03-REJ-07-N2 | 2 | UC-03 | turn on the lamp, please | `null` |
| UC03-REJ-08 | 1 | UC-03 | turn it on | `null` |
| UC03-REJ-08-N1 | 2 | UC-03 | hey assistant, turn it on | `null` |
| UC03-REJ-08-N2 | 2 | UC-03 | turn it on, please | `null` |
| UC03-REJ-09 | 1 | UC-03 | turn kitchen lamp on off | `null` |
| UC03-REJ-09-N1 | 2 | UC-03 | hey assistant, turn kitchen lamp on off | `null` |
| UC03-REJ-09-N2 | 2 | UC-03 | turn kitchen lamp on off, please | `null` |
| UC03-REJ-10 | 1 | UC-03 | turn on kitchen lamp not bedroom lamp | `null` |
| UC03-REJ-10-N1 | 2 | UC-03 | hey assistant, turn on kitchen lamp not bedroom lamp | `null` |
| UC03-REJ-10-N2 | 2 | UC-03 | turn on kitchen lamp not bedroom lamp, please | `null` |
| UC03-REJ-11 | 1 | UC-03 | turn on kitchen lamp two | `null` |
| UC03-REJ-11-N1 | 2 | UC-03 | hey assistant, turn on kitchen lamp two | `null` |
| UC03-REJ-11-N2 | 2 | UC-03 | turn on kitchen lamp two, please | `null` |
| UC03-REJ-12 | 1 | UC-03 | turn on lamp two three | `null` |
| UC03-REJ-12-N1 | 2 | UC-03 | hey assistant, turn on lamp two three | `null` |
| UC03-REJ-12-N2 | 2 | UC-03 | turn on lamp two three, please | `null` |
| UC03-REJ-13 | 1 | UC-03 | turn on bathroom lamp | `null` |
| UC03-REJ-13-N1 | 2 | UC-03 | hey assistant, turn on bathroom lamp | `null` |
| UC03-REJ-13-N2 | 2 | UC-03 | turn on bathroom lamp, please | `null` |
| UC03-REJ-14 | 1 | UC-03 | turn off garden light | `null` |
| UC03-REJ-14-N1 | 2 | UC-03 | hey assistant, turn off garden light | `null` |
| UC03-REJ-14-N2 | 2 | UC-03 | turn off garden light, please | `null` |
| UC03-REJ-15 | 1 | UC-03 | switch on kitchen lamp | `null` |
| UC03-REJ-15-N1 | 2 | UC-03 | hey assistant, switch on kitchen lamp | `null` |
| UC03-REJ-15-N2 | 2 | UC-03 | switch on kitchen lamp, please | `null` |
| UC03-REJ-16 | 1 | UC-03 | turn off all lights | `null` |
| UC03-REJ-16-N1 | 2 | UC-03 | hey assistant, turn off all lights | `null` |
| UC03-REJ-16-N2 | 2 | UC-03 | turn off all lights, please | `null` |
| UC03-REJ-17 | 1 | UC-03 | turn on bedroom lights | `null` |
| UC03-REJ-17-N1 | 2 | UC-03 | hey assistant, turn on bedroom lights | `null` |
| UC03-REJ-17-N2 | 2 | UC-03 | turn on bedroom lights, please | `null` |
| UC03-REJ-18 | 1 | UC-03 | turn on kitchen lamp please | `null` |
| UC03-REJ-18-N1 | 2 | UC-03 | hey assistant, turn on kitchen lamp please | `null` |
| UC03-REJ-18-N2 | 2 | UC-03 | turn on kitchen lamp please, please | `null` |
| UC03-REJ-19 | 1 | UC-03 | turn on ceiling fan | `null` |
| UC03-REJ-19-N1 | 2 | UC-03 | hey assistant, turn on ceiling fan | `null` |
| UC03-REJ-19-N2 | 2 | UC-03 | turn on ceiling fan, please | `null` |
| UC03-REJ-20 | 1 | UC-03 | dim kitchen lamp | `null` |
| UC03-REJ-20-N1 | 2 | UC-03 | hey assistant, dim kitchen lamp | `null` |
| UC03-REJ-20-N2 | 2 | UC-03 | dim kitchen lamp, please | `null` |
| UC04-REJ-01 | 1 | UC-04 | do not set kitchen lamp to seventy three percent | `null` |
| UC04-REJ-01-N1 | 2 | UC-04 | hey assistant, do not set kitchen lamp to seventy three percent | `null` |
| UC04-REJ-01-N2 | 2 | UC-04 | do not set kitchen lamp to seventy three percent, please | `null` |
| UC04-REJ-02 | 1 | UC-04 | set kitchen lamp to seventy three percent eighty | `null` |
| UC04-REJ-02-N1 | 2 | UC-04 | hey assistant, set kitchen lamp to seventy three percent eighty | `null` |
| UC04-REJ-02-N2 | 2 | UC-04 | set kitchen lamp to seventy three percent eighty, please | `null` |
| UC04-REJ-03 | 1 | UC-04 | set kitchen lamp to one hundred one percent | `null` |
| UC04-REJ-03-N1 | 2 | UC-04 | hey assistant, set kitchen lamp to one hundred one percent | `null` |
| UC04-REJ-03-N2 | 2 | UC-04 | set kitchen lamp to one hundred one percent, please | `null` |
| UC04-REJ-04 | 1 | UC-04 | set kitchen lamp to minus one percent | `null` |
| UC04-REJ-04-N1 | 2 | UC-04 | hey assistant, set kitchen lamp to minus one percent | `null` |
| UC04-REJ-04-N2 | 2 | UC-04 | set kitchen lamp to minus one percent, please | `null` |
| UC04-REJ-05 | 1 | UC-04 | set kitchen lamp to seventy point three percent | `null` |
| UC04-REJ-05-N1 | 2 | UC-04 | hey assistant, set kitchen lamp to seventy point three percent | `null` |
| UC04-REJ-05-N2 | 2 | UC-04 | set kitchen lamp to seventy point three percent, please | `null` |
| UC04-REJ-06 | 1 | UC-04 | set kitchen lamp to seventy three | `null` |
| UC04-REJ-06-N1 | 2 | UC-04 | hey assistant, set kitchen lamp to seventy three | `null` |
| UC04-REJ-06-N2 | 2 | UC-04 | set kitchen lamp to seventy three, please | `null` |
| UC04-REJ-07 | 1 | UC-04 | set kitchen lamp to percent | `null` |
| UC04-REJ-07-N1 | 2 | UC-04 | hey assistant, set kitchen lamp to percent | `null` |
| UC04-REJ-07-N2 | 2 | UC-04 | set kitchen lamp to percent, please | `null` |
| UC04-REJ-08 | 1 | UC-04 | set kitchen lamp to seventy three percent and call alex | `null` |
| UC04-REJ-08-N1 | 2 | UC-04 | hey assistant, set kitchen lamp to seventy three percent and call alex | `null` |
| UC04-REJ-08-N2 | 2 | UC-04 | set kitchen lamp to seventy three percent and call alex, please | `null` |
| UC04-REJ-09 | 1 | UC-04 | increase kitchen lamp to five percent by ten percent | `null` |
| UC04-REJ-09-N1 | 2 | UC-04 | hey assistant, increase kitchen lamp to five percent by ten percent | `null` |
| UC04-REJ-09-N2 | 2 | UC-04 | increase kitchen lamp to five percent by ten percent, please | `null` |
| UC04-REJ-10 | 1 | UC-04 | set lamp two to three percent four | `null` |
| UC04-REJ-10-N1 | 2 | UC-04 | hey assistant, set lamp two to three percent four | `null` |
| UC04-REJ-10-N2 | 2 | UC-04 | set lamp two to three percent four, please | `null` |
| UC04-REJ-11 | 1 | UC-04 | set kitchen lamp to red blue | `null` |
| UC04-REJ-11-N1 | 2 | UC-04 | hey assistant, set kitchen lamp to red blue | `null` |
| UC04-REJ-11-N2 | 2 | UC-04 | set kitchen lamp to red blue, please | `null` |
| UC04-REJ-12 | 1 | UC-04 | set lamp to red | `null` |
| UC04-REJ-12-N1 | 2 | UC-04 | hey assistant, set lamp to red | `null` |
| UC04-REJ-12-N2 | 2 | UC-04 | set lamp to red, please | `null` |
| UC04-REJ-13 | 1 | UC-04 | set bedroom lamp to red | `null` |
| UC04-REJ-13-N1 | 2 | UC-04 | hey assistant, set bedroom lamp to red | `null` |
| UC04-REJ-13-N2 | 2 | UC-04 | set bedroom lamp to red, please | `null` |
| UC04-REJ-14 | 1 | UC-04 | set lamp two to blue | `null` |
| UC04-REJ-14-N1 | 2 | UC-04 | hey assistant, set lamp two to blue | `null` |
| UC04-REJ-14-N2 | 2 | UC-04 | set lamp two to blue, please | `null` |
| UC04-REJ-15 | 1 | UC-04 | set kitchen lamp to orange | `null` |
| UC04-REJ-15-N1 | 2 | UC-04 | hey assistant, set kitchen lamp to orange | `null` |
| UC04-REJ-15-N2 | 2 | UC-04 | set kitchen lamp to orange, please | `null` |
| UC04-REJ-16 | 1 | UC-04 | set kitchen lamp to cyan | `null` |
| UC04-REJ-16-N1 | 2 | UC-04 | hey assistant, set kitchen lamp to cyan | `null` |
| UC04-REJ-16-N2 | 2 | UC-04 | set kitchen lamp to cyan, please | `null` |
| UC04-REJ-17 | 1 | UC-04 | increase kitchen lamp by twenty percent | `null` |
| UC04-REJ-17-N1 | 2 | UC-04 | hey assistant, increase kitchen lamp by twenty percent | `null` |
| UC04-REJ-17-N2 | 2 | UC-04 | increase kitchen lamp by twenty percent, please | `null` |
| UC04-REJ-18 | 1 | UC-04 | set garden lamp to fifty percent | `null` |
| UC04-REJ-18-N1 | 2 | UC-04 | hey assistant, set garden lamp to fifty percent | `null` |
| UC04-REJ-18-N2 | 2 | UC-04 | set garden lamp to fifty percent, please | `null` |
| UC04-REJ-19 | 1 | UC-04 | set kitchen lamp to half brightness | `null` |
| UC04-REJ-19-N1 | 2 | UC-04 | hey assistant, set kitchen lamp to half brightness | `null` |
| UC04-REJ-19-N2 | 2 | UC-04 | set kitchen lamp to half brightness, please | `null` |
| UC04-REJ-20 | 1 | UC-04 | make kitchen lamp warmer | `null` |
| UC04-REJ-20-N1 | 2 | UC-04 | hey assistant, make kitchen lamp warmer | `null` |
| UC04-REJ-20-N2 | 2 | UC-04 | make kitchen lamp warmer, please | `null` |
| UC05-REJ-01 | 1 | UC-05 | do not set a timer for five minutes | `null` |
| UC05-REJ-01-N1 | 2 | UC-05 | hey assistant, do not set a timer for five minutes | `null` |
| UC05-REJ-01-N2 | 2 | UC-05 | do not set a timer for five minutes, please | `null` |
| UC05-REJ-02 | 1 | UC-05 | set a timer for five minutes and call alex | `null` |
| UC05-REJ-02-N1 | 2 | UC-05 | hey assistant, set a timer for five minutes and call alex | `null` |
| UC05-REJ-02-N2 | 2 | UC-05 | set a timer for five minutes and call alex, please | `null` |
| UC05-REJ-03 | 1 | UC-05 | set a timer for | `null` |
| UC05-REJ-03-N1 | 2 | UC-05 | hey assistant, set a timer for | `null` |
| UC05-REJ-03-N2 | 2 | UC-05 | set a timer for, please | `null` |
| UC05-REJ-04 | 1 | UC-05 | set a timer | `null` |
| UC05-REJ-04-N1 | 2 | UC-05 | hey assistant, set a timer | `null` |
| UC05-REJ-04-N2 | 2 | UC-05 | set a timer, please | `null` |
| UC05-REJ-05 | 1 | UC-05 | set a timer for five | `null` |
| UC05-REJ-05-N1 | 2 | UC-05 | hey assistant, set a timer for five | `null` |
| UC05-REJ-05-N2 | 2 | UC-05 | set a timer for five, please | `null` |
| UC05-REJ-06 | 1 | UC-05 | set a timer for minutes | `null` |
| UC05-REJ-06-N1 | 2 | UC-05 | hey assistant, set a timer for minutes | `null` |
| UC05-REJ-06-N2 | 2 | UC-05 | set a timer for minutes, please | `null` |
| UC05-REJ-07 | 1 | UC-05 | set a timer for minus five minutes | `null` |
| UC05-REJ-07-N1 | 2 | UC-05 | hey assistant, set a timer for minus five minutes | `null` |
| UC05-REJ-07-N2 | 2 | UC-05 | set a timer for minus five minutes, please | `null` |
| UC05-REJ-08 | 1 | UC-05 | set a timer for zero minutes | `null` |
| UC05-REJ-08-N1 | 2 | UC-05 | hey assistant, set a timer for zero minutes | `null` |
| UC05-REJ-08-N2 | 2 | UC-05 | set a timer for zero minutes, please | `null` |
| UC05-REJ-09 | 1 | UC-05 | set a timer for five minutes ten | `null` |
| UC05-REJ-09-N1 | 2 | UC-05 | hey assistant, set a timer for five minutes ten | `null` |
| UC05-REJ-09-N2 | 2 | UC-05 | set a timer for five minutes ten, please | `null` |
| UC05-REJ-10 | 1 | UC-05 | set a timer for five minutes not ten minutes | `null` |
| UC05-REJ-10-N1 | 2 | UC-05 | hey assistant, set a timer for five minutes not ten minutes | `null` |
| UC05-REJ-10-N2 | 2 | UC-05 | set a timer for five minutes not ten minutes, please | `null` |
| UC05-REJ-11 | 1 | UC-05 | cancel | `null` |
| UC05-REJ-11-N1 | 2 | UC-05 | hey assistant, cancel | `null` |
| UC05-REJ-11-N2 | 2 | UC-05 | cancel, please | `null` |
| UC05-REJ-12 | 1 | UC-05 | cancel timer and start a timer for one hour | `null` |
| UC05-REJ-12-N1 | 2 | UC-05 | hey assistant, cancel timer and start a timer for one hour | `null` |
| UC05-REJ-12-N2 | 2 | UC-05 | cancel timer and start a timer for one hour, please | `null` |
| UC05-REJ-13 | 1 | UC-05 | set a timer for three minutes | `null` |
| UC05-REJ-13-N1 | 2 | UC-05 | hey assistant, set a timer for three minutes | `null` |
| UC05-REJ-13-N2 | 2 | UC-05 | set a timer for three minutes, please | `null` |
| UC05-REJ-14 | 1 | UC-05 | set a timer for thirty seconds | `null` |
| UC05-REJ-14-N1 | 2 | UC-05 | hey assistant, set a timer for thirty seconds | `null` |
| UC05-REJ-14-N2 | 2 | UC-05 | set a timer for thirty seconds, please | `null` |
| UC05-REJ-15 | 1 | UC-05 | set a timer for two hours | `null` |
| UC05-REJ-15-N1 | 2 | UC-05 | hey assistant, set a timer for two hours | `null` |
| UC05-REJ-15-N2 | 2 | UC-05 | set a timer for two hours, please | `null` |
| UC05-REJ-16 | 1 | UC-05 | set a timer for ninety seconds | `null` |
| UC05-REJ-16-N1 | 2 | UC-05 | hey assistant, set a timer for ninety seconds | `null` |
| UC05-REJ-16-N2 | 2 | UC-05 | set a timer for ninety seconds, please | `null` |
| UC05-REJ-17 | 1 | UC-05 | pause timer | `null` |
| UC05-REJ-17-N1 | 2 | UC-05 | hey assistant, pause timer | `null` |
| UC05-REJ-17-N2 | 2 | UC-05 | pause timer, please | `null` |
| UC05-REJ-18 | 1 | UC-05 | resume timer | `null` |
| UC05-REJ-18-N1 | 2 | UC-05 | hey assistant, resume timer | `null` |
| UC05-REJ-18-N2 | 2 | UC-05 | resume timer, please | `null` |
| UC05-REJ-19 | 1 | UC-05 | add five minutes to timer | `null` |
| UC05-REJ-19-N1 | 2 | UC-05 | hey assistant, add five minutes to timer | `null` |
| UC05-REJ-19-N2 | 2 | UC-05 | add five minutes to timer, please | `null` |
| UC05-REJ-20 | 1 | UC-05 | cancel the second timer | `null` |
| UC05-REJ-20-N1 | 2 | UC-05 | hey assistant, cancel the second timer | `null` |
| UC05-REJ-20-N2 | 2 | UC-05 | cancel the second timer, please | `null` |
| UC06-REJ-01 | 1 | UC-06 | do not set an alarm for seven a m | `null` |
| UC06-REJ-01-N1 | 2 | UC-06 | hey assistant, do not set an alarm for seven a m | `null` |
| UC06-REJ-01-N2 | 2 | UC-06 | do not set an alarm for seven a m, please | `null` |
| UC06-REJ-02 | 1 | UC-06 | set an alarm for seven a m and call alex | `null` |
| UC06-REJ-02-N1 | 2 | UC-06 | hey assistant, set an alarm for seven a m and call alex | `null` |
| UC06-REJ-02-N2 | 2 | UC-06 | set an alarm for seven a m and call alex, please | `null` |
| UC06-REJ-03 | 1 | UC-06 | set an alarm for | `null` |
| UC06-REJ-03-N1 | 2 | UC-06 | hey assistant, set an alarm for | `null` |
| UC06-REJ-03-N2 | 2 | UC-06 | set an alarm for, please | `null` |
| UC06-REJ-04 | 1 | UC-06 | set an alarm for seven | `null` |
| UC06-REJ-04-N1 | 2 | UC-06 | hey assistant, set an alarm for seven | `null` |
| UC06-REJ-04-N2 | 2 | UC-06 | set an alarm for seven, please | `null` |
| UC06-REJ-05 | 1 | UC-06 | set an alarm for seven thirty | `null` |
| UC06-REJ-05-N1 | 2 | UC-06 | hey assistant, set an alarm for seven thirty | `null` |
| UC06-REJ-05-N2 | 2 | UC-06 | set an alarm for seven thirty, please | `null` |
| UC06-REJ-06 | 1 | UC-06 | set an alarm for twenty five a m | `null` |
| UC06-REJ-06-N1 | 2 | UC-06 | hey assistant, set an alarm for twenty five a m | `null` |
| UC06-REJ-06-N2 | 2 | UC-06 | set an alarm for twenty five a m, please | `null` |
| UC06-REJ-07 | 1 | UC-06 | set an alarm for seven sixty a m | `null` |
| UC06-REJ-07-N1 | 2 | UC-06 | hey assistant, set an alarm for seven sixty a m | `null` |
| UC06-REJ-07-N2 | 2 | UC-06 | set an alarm for seven sixty a m, please | `null` |
| UC06-REJ-08 | 1 | UC-06 | set an alarm for six a m p m | `null` |
| UC06-REJ-08-N1 | 2 | UC-06 | hey assistant, set an alarm for six a m p m | `null` |
| UC06-REJ-08-N2 | 2 | UC-06 | set an alarm for six a m p m, please | `null` |
| UC06-REJ-09 | 1 | UC-06 | set an alarm for seven a m eight | `null` |
| UC06-REJ-09-N1 | 2 | UC-06 | hey assistant, set an alarm for seven a m eight | `null` |
| UC06-REJ-09-N2 | 2 | UC-06 | set an alarm for seven a m eight, please | `null` |
| UC06-REJ-10 | 1 | UC-06 | set an alarm for six a m not six p m | `null` |
| UC06-REJ-10-N1 | 2 | UC-06 | hey assistant, set an alarm for six a m not six p m | `null` |
| UC06-REJ-10-N2 | 2 | UC-06 | set an alarm for six a m not six p m, please | `null` |
| UC06-REJ-11 | 1 | UC-06 | wake me | `null` |
| UC06-REJ-11-N1 | 2 | UC-06 | hey assistant, wake me | `null` |
| UC06-REJ-11-N2 | 2 | UC-06 | wake me, please | `null` |
| UC06-REJ-12 | 1 | UC-06 | cancel alarm and set an alarm for six a m | `null` |
| UC06-REJ-12-N1 | 2 | UC-06 | hey assistant, cancel alarm and set an alarm for six a m | `null` |
| UC06-REJ-12-N2 | 2 | UC-06 | cancel alarm and set an alarm for six a m, please | `null` |
| UC06-REJ-13 | 1 | UC-06 | set an alarm for seven fifteen a m | `null` |
| UC06-REJ-13-N1 | 2 | UC-06 | hey assistant, set an alarm for seven fifteen a m | `null` |
| UC06-REJ-13-N2 | 2 | UC-06 | set an alarm for seven fifteen a m, please | `null` |
| UC06-REJ-14 | 1 | UC-06 | set an alarm for nine a m | `null` |
| UC06-REJ-14-N1 | 2 | UC-06 | hey assistant, set an alarm for nine a m | `null` |
| UC06-REJ-14-N2 | 2 | UC-06 | set an alarm for nine a m, please | `null` |
| UC06-REJ-15 | 1 | UC-06 | set an alarm for seven p m | `null` |
| UC06-REJ-15-N1 | 2 | UC-06 | hey assistant, set an alarm for seven p m | `null` |
| UC06-REJ-15-N2 | 2 | UC-06 | set an alarm for seven p m, please | `null` |
| UC06-REJ-16 | 1 | UC-06 | set an alarm for midnight | `null` |
| UC06-REJ-16-N1 | 2 | UC-06 | hey assistant, set an alarm for midnight | `null` |
| UC06-REJ-16-N2 | 2 | UC-06 | set an alarm for midnight, please | `null` |
| UC06-REJ-17 | 1 | UC-06 | set an alarm for noon | `null` |
| UC06-REJ-17-N1 | 2 | UC-06 | hey assistant, set an alarm for noon | `null` |
| UC06-REJ-17-N2 | 2 | UC-06 | set an alarm for noon, please | `null` |
| UC06-REJ-18 | 1 | UC-06 | set an alarm for seven a m every day | `null` |
| UC06-REJ-18-N1 | 2 | UC-06 | hey assistant, set an alarm for seven a m every day | `null` |
| UC06-REJ-18-N2 | 2 | UC-06 | set an alarm for seven a m every day, please | `null` |
| UC06-REJ-19 | 1 | UC-06 | snooze alarm | `null` |
| UC06-REJ-19-N1 | 2 | UC-06 | hey assistant, snooze alarm | `null` |
| UC06-REJ-19-N2 | 2 | UC-06 | snooze alarm, please | `null` |
| UC06-REJ-20 | 1 | UC-06 | cancel my weekday alarm | `null` |
| UC06-REJ-20-N1 | 2 | UC-06 | hey assistant, cancel my weekday alarm | `null` |
| UC06-REJ-20-N2 | 2 | UC-06 | cancel my weekday alarm, please | `null` |
| UC07-REJ-01 | 1 | UC-07 | do not set thermostat to twenty degrees celsius | `null` |
| UC07-REJ-01-N1 | 2 | UC-07 | hey assistant, do not set thermostat to twenty degrees celsius | `null` |
| UC07-REJ-01-N2 | 2 | UC-07 | do not set thermostat to twenty degrees celsius, please | `null` |
| UC07-REJ-02 | 1 | UC-07 | set thermostat to twenty degrees celsius and call alex | `null` |
| UC07-REJ-02-N1 | 2 | UC-07 | hey assistant, set thermostat to twenty degrees celsius and call alex | `null` |
| UC07-REJ-02-N2 | 2 | UC-07 | set thermostat to twenty degrees celsius and call alex, please | `null` |
| UC07-REJ-03 | 1 | UC-07 | set thermostat to | `null` |
| UC07-REJ-03-N1 | 2 | UC-07 | hey assistant, set thermostat to | `null` |
| UC07-REJ-03-N2 | 2 | UC-07 | set thermostat to, please | `null` |
| UC07-REJ-04 | 1 | UC-07 | set thermostat to twenty | `null` |
| UC07-REJ-04-N1 | 2 | UC-07 | hey assistant, set thermostat to twenty | `null` |
| UC07-REJ-04-N2 | 2 | UC-07 | set thermostat to twenty, please | `null` |
| UC07-REJ-05 | 1 | UC-07 | set thermostat to degrees celsius | `null` |
| UC07-REJ-05-N1 | 2 | UC-07 | hey assistant, set thermostat to degrees celsius | `null` |
| UC07-REJ-05-N2 | 2 | UC-07 | set thermostat to degrees celsius, please | `null` |
| UC07-REJ-06 | 1 | UC-07 | set thermostat to twenty degrees celsius thirty | `null` |
| UC07-REJ-06-N1 | 2 | UC-07 | hey assistant, set thermostat to twenty degrees celsius thirty | `null` |
| UC07-REJ-06-N2 | 2 | UC-07 | set thermostat to twenty degrees celsius thirty, please | `null` |
| UC07-REJ-07 | 1 | UC-07 | set thermostat to twenty degrees fahrenheit celsius | `null` |
| UC07-REJ-07-N1 | 2 | UC-07 | hey assistant, set thermostat to twenty degrees fahrenheit celsius | `null` |
| UC07-REJ-07-N2 | 2 | UC-07 | set thermostat to twenty degrees fahrenheit celsius, please | `null` |
| UC07-REJ-08 | 1 | UC-07 | set thermostat to twenty point five degrees celsius | `null` |
| UC07-REJ-08-N1 | 2 | UC-07 | hey assistant, set thermostat to twenty point five degrees celsius | `null` |
| UC07-REJ-08-N2 | 2 | UC-07 | set thermostat to twenty point five degrees celsius, please | `null` |
| UC07-REJ-09 | 1 | UC-07 | set thermostat to minus twenty degrees celsius | `null` |
| UC07-REJ-09-N1 | 2 | UC-07 | hey assistant, set thermostat to minus twenty degrees celsius | `null` |
| UC07-REJ-09-N2 | 2 | UC-07 | set thermostat to minus twenty degrees celsius, please | `null` |
| UC07-REJ-10 | 1 | UC-07 | increase thermostat by degrees celsius | `null` |
| UC07-REJ-10-N1 | 2 | UC-07 | hey assistant, increase thermostat by degrees celsius | `null` |
| UC07-REJ-10-N2 | 2 | UC-07 | increase thermostat by degrees celsius, please | `null` |
| UC07-REJ-11 | 1 | UC-07 | increase thermostat to one by two degrees celsius | `null` |
| UC07-REJ-11-N1 | 2 | UC-07 | hey assistant, increase thermostat to one by two degrees celsius | `null` |
| UC07-REJ-11-N2 | 2 | UC-07 | increase thermostat to one by two degrees celsius, please | `null` |
| UC07-REJ-12 | 1 | UC-07 | set thermostat to twenty degrees celsius not thirty | `null` |
| UC07-REJ-12-N1 | 2 | UC-07 | hey assistant, set thermostat to twenty degrees celsius not thirty | `null` |
| UC07-REJ-12-N2 | 2 | UC-07 | set thermostat to twenty degrees celsius not thirty, please | `null` |
| UC07-REJ-13 | 1 | UC-07 | set thermostat to fifteen degrees celsius | `null` |
| UC07-REJ-13-N1 | 2 | UC-07 | hey assistant, set thermostat to fifteen degrees celsius | `null` |
| UC07-REJ-13-N2 | 2 | UC-07 | set thermostat to fifteen degrees celsius, please | `null` |
| UC07-REJ-14 | 1 | UC-07 | set thermostat to thirty one degrees celsius | `null` |
| UC07-REJ-14-N1 | 2 | UC-07 | hey assistant, set thermostat to thirty one degrees celsius | `null` |
| UC07-REJ-14-N2 | 2 | UC-07 | set thermostat to thirty one degrees celsius, please | `null` |
| UC07-REJ-15 | 1 | UC-07 | set thermostat to seventy degrees fahrenheit | `null` |
| UC07-REJ-15-N1 | 2 | UC-07 | hey assistant, set thermostat to seventy degrees fahrenheit | `null` |
| UC07-REJ-15-N2 | 2 | UC-07 | set thermostat to seventy degrees fahrenheit, please | `null` |
| UC07-REJ-16 | 1 | UC-07 | increase thermostat by three degrees celsius | `null` |
| UC07-REJ-16-N1 | 2 | UC-07 | hey assistant, increase thermostat by three degrees celsius | `null` |
| UC07-REJ-16-N2 | 2 | UC-07 | increase thermostat by three degrees celsius, please | `null` |
| UC07-REJ-17 | 1 | UC-07 | set bedroom thermostat to twenty degrees celsius | `null` |
| UC07-REJ-17-N1 | 2 | UC-07 | hey assistant, set bedroom thermostat to twenty degrees celsius | `null` |
| UC07-REJ-17-N2 | 2 | UC-07 | set bedroom thermostat to twenty degrees celsius, please | `null` |
| UC07-REJ-18 | 1 | UC-07 | make the room warmer | `null` |
| UC07-REJ-18-N1 | 2 | UC-07 | hey assistant, make the room warmer | `null` |
| UC07-REJ-18-N2 | 2 | UC-07 | make the room warmer, please | `null` |
| UC07-REJ-19 | 1 | UC-07 | turn on air conditioning | `null` |
| UC07-REJ-19-N1 | 2 | UC-07 | hey assistant, turn on air conditioning | `null` |
| UC07-REJ-19-N2 | 2 | UC-07 | turn on air conditioning, please | `null` |
| UC07-REJ-20 | 1 | UC-07 | set thermostat to eco mode | `null` |
| UC07-REJ-20-N1 | 2 | UC-07 | hey assistant, set thermostat to eco mode | `null` |
| UC07-REJ-20-N2 | 2 | UC-07 | set thermostat to eco mode, please | `null` |
| UC08-REJ-01 | 1 | UC-08 | do not pause music | `null` |
| UC08-REJ-01-N1 | 2 | UC-08 | hey assistant, do not pause music | `null` |
| UC08-REJ-01-N2 | 2 | UC-08 | do not pause music, please | `null` |
| UC08-REJ-02 | 1 | UC-08 | do not set volume to seventy three percent | `null` |
| UC08-REJ-02-N1 | 2 | UC-08 | hey assistant, do not set volume to seventy three percent | `null` |
| UC08-REJ-02-N2 | 2 | UC-08 | do not set volume to seventy three percent, please | `null` |
| UC08-REJ-03 | 1 | UC-08 | set volume to one hundred one percent | `null` |
| UC08-REJ-03-N1 | 2 | UC-08 | hey assistant, set volume to one hundred one percent | `null` |
| UC08-REJ-03-N2 | 2 | UC-08 | set volume to one hundred one percent, please | `null` |
| UC08-REJ-04 | 1 | UC-08 | set volume to minus one percent | `null` |
| UC08-REJ-04-N1 | 2 | UC-08 | hey assistant, set volume to minus one percent | `null` |
| UC08-REJ-04-N2 | 2 | UC-08 | set volume to minus one percent, please | `null` |
| UC08-REJ-05 | 1 | UC-08 | set volume to seventy point three percent | `null` |
| UC08-REJ-05-N1 | 2 | UC-08 | hey assistant, set volume to seventy point three percent | `null` |
| UC08-REJ-05-N2 | 2 | UC-08 | set volume to seventy point three percent, please | `null` |
| UC08-REJ-06 | 1 | UC-08 | set volume to seventy three | `null` |
| UC08-REJ-06-N1 | 2 | UC-08 | hey assistant, set volume to seventy three | `null` |
| UC08-REJ-06-N2 | 2 | UC-08 | set volume to seventy three, please | `null` |
| UC08-REJ-07 | 1 | UC-08 | set volume to percent | `null` |
| UC08-REJ-07-N1 | 2 | UC-08 | hey assistant, set volume to percent | `null` |
| UC08-REJ-07-N2 | 2 | UC-08 | set volume to percent, please | `null` |
| UC08-REJ-08 | 1 | UC-08 | set volume to seventy three percent eighty | `null` |
| UC08-REJ-08-N1 | 2 | UC-08 | hey assistant, set volume to seventy three percent eighty | `null` |
| UC08-REJ-08-N2 | 2 | UC-08 | set volume to seventy three percent eighty, please | `null` |
| UC08-REJ-09 | 1 | UC-08 | set volume to seventy three percent and call alex | `null` |
| UC08-REJ-09-N1 | 2 | UC-08 | hey assistant, set volume to seventy three percent and call alex | `null` |
| UC08-REJ-09-N2 | 2 | UC-08 | set volume to seventy three percent and call alex, please | `null` |
| UC08-REJ-10 | 1 | UC-08 | pause music and resume music | `null` |
| UC08-REJ-10-N1 | 2 | UC-08 | hey assistant, pause music and resume music | `null` |
| UC08-REJ-10-N2 | 2 | UC-08 | pause music and resume music, please | `null` |
| UC08-REJ-11 | 1 | UC-08 | set volume to seventy three percent not thirty seven percent | `null` |
| UC08-REJ-11-N1 | 2 | UC-08 | hey assistant, set volume to seventy three percent not thirty seven percent | `null` |
| UC08-REJ-11-N2 | 2 | UC-08 | set volume to seventy three percent not thirty seven percent, please | `null` |
| UC08-REJ-12 | 1 | UC-08 | volume up down | `null` |
| UC08-REJ-12-N1 | 2 | UC-08 | hey assistant, volume up down | `null` |
| UC08-REJ-12-N2 | 2 | UC-08 | volume up down, please | `null` |
| UC08-REJ-13 | 1 | UC-08 | increase volume by twenty percent | `null` |
| UC08-REJ-13-N1 | 2 | UC-08 | hey assistant, increase volume by twenty percent | `null` |
| UC08-REJ-13-N2 | 2 | UC-08 | increase volume by twenty percent, please | `null` |
| UC08-REJ-14 | 1 | UC-08 | set volume to half | `null` |
| UC08-REJ-14-N1 | 2 | UC-08 | hey assistant, set volume to half | `null` |
| UC08-REJ-14-N2 | 2 | UC-08 | set volume to half, please | `null` |
| UC08-REJ-15 | 1 | UC-08 | set bedroom speaker volume to fifty percent | `null` |
| UC08-REJ-15-N1 | 2 | UC-08 | hey assistant, set bedroom speaker volume to fifty percent | `null` |
| UC08-REJ-15-N2 | 2 | UC-08 | set bedroom speaker volume to fifty percent, please | `null` |
| UC08-REJ-16 | 1 | UC-08 | play the next album | `null` |
| UC08-REJ-16-N1 | 2 | UC-08 | hey assistant, play the next album | `null` |
| UC08-REJ-16-N2 | 2 | UC-08 | play the next album, please | `null` |
| UC08-REJ-17 | 1 | UC-08 | rewind ten seconds | `null` |
| UC08-REJ-17-N1 | 2 | UC-08 | hey assistant, rewind ten seconds | `null` |
| UC08-REJ-17-N2 | 2 | UC-08 | rewind ten seconds, please | `null` |
| UC08-REJ-18 | 1 | UC-08 | fast forward music | `null` |
| UC08-REJ-18-N1 | 2 | UC-08 | hey assistant, fast forward music | `null` |
| UC08-REJ-18-N2 | 2 | UC-08 | fast forward music, please | `null` |
| UC08-REJ-19 | 1 | UC-08 | repeat this track | `null` |
| UC08-REJ-19-N1 | 2 | UC-08 | hey assistant, repeat this track | `null` |
| UC08-REJ-19-N2 | 2 | UC-08 | repeat this track, please | `null` |
| UC08-REJ-20 | 1 | UC-08 | resume the music | `null` |
| UC08-REJ-20-N1 | 2 | UC-08 | hey assistant, resume the music | `null` |
| UC08-REJ-20-N2 | 2 | UC-08 | resume the music, please | `null` |
| UC09-REJ-01 | 1 | UC-09 | do not remind me to drink water in ten minutes | `null` |
| UC09-REJ-01-N1 | 2 | UC-09 | hey assistant, do not remind me to drink water in ten minutes | `null` |
| UC09-REJ-01-N2 | 2 | UC-09 | do not remind me to drink water in ten minutes, please | `null` |
| UC09-REJ-02 | 1 | UC-09 | do not add milk to my shopping list | `null` |
| UC09-REJ-02-N1 | 2 | UC-09 | hey assistant, do not add milk to my shopping list | `null` |
| UC09-REJ-02-N2 | 2 | UC-09 | do not add milk to my shopping list, please | `null` |
| UC09-REJ-03 | 1 | UC-09 | remind me to | `null` |
| UC09-REJ-03-N1 | 2 | UC-09 | hey assistant, remind me to | `null` |
| UC09-REJ-03-N2 | 2 | UC-09 | remind me to, please | `null` |
| UC09-REJ-04 | 1 | UC-09 | remind me to drink water in | `null` |
| UC09-REJ-04-N1 | 2 | UC-09 | hey assistant, remind me to drink water in | `null` |
| UC09-REJ-04-N2 | 2 | UC-09 | remind me to drink water in, please | `null` |
| UC09-REJ-05 | 1 | UC-09 | add to my shopping list | `null` |
| UC09-REJ-05-N1 | 2 | UC-09 | hey assistant, add to my shopping list | `null` |
| UC09-REJ-05-N2 | 2 | UC-09 | add to my shopping list, please | `null` |
| UC09-REJ-06 | 1 | UC-09 | add milk to | `null` |
| UC09-REJ-06-N1 | 2 | UC-09 | hey assistant, add milk to | `null` |
| UC09-REJ-06-N2 | 2 | UC-09 | add milk to, please | `null` |
| UC09-REJ-07 | 1 | UC-09 | add milk to my shopping list and call alex | `null` |
| UC09-REJ-07-N1 | 2 | UC-09 | hey assistant, add milk to my shopping list and call alex | `null` |
| UC09-REJ-07-N2 | 2 | UC-09 | add milk to my shopping list and call alex, please | `null` |
| UC09-REJ-08 | 1 | UC-09 | remove milk from my shopping list and add bread to my shopping list | `null` |
| UC09-REJ-08-N1 | 2 | UC-09 | hey assistant, remove milk from my shopping list and add bread to my shopping list | `null` |
| UC09-REJ-08-N2 | 2 | UC-09 | remove milk from my shopping list and add bread to my shopping list, please | `null` |
| UC09-REJ-09 | 1 | UC-09 | remind me to drink water in minus ten minutes | `null` |
| UC09-REJ-09-N1 | 2 | UC-09 | hey assistant, remind me to drink water in minus ten minutes | `null` |
| UC09-REJ-09-N2 | 2 | UC-09 | remind me to drink water in minus ten minutes, please | `null` |
| UC09-REJ-10 | 1 | UC-09 | remind me to drink water in ten minutes five | `null` |
| UC09-REJ-10-N1 | 2 | UC-09 | hey assistant, remind me to drink water in ten minutes five | `null` |
| UC09-REJ-10-N2 | 2 | UC-09 | remind me to drink water in ten minutes five, please | `null` |
| UC09-REJ-11 | 1 | UC-09 | add milk bread to my shopping list | `null` |
| UC09-REJ-11-N1 | 2 | UC-09 | hey assistant, add milk bread to my shopping list | `null` |
| UC09-REJ-11-N2 | 2 | UC-09 | add milk bread to my shopping list, please | `null` |
| UC09-REJ-12 | 1 | UC-09 | cancel my reminder | `null` |
| UC09-REJ-12-N1 | 2 | UC-09 | hey assistant, cancel my reminder | `null` |
| UC09-REJ-12-N2 | 2 | UC-09 | cancel my reminder, please | `null` |
| UC09-REJ-13 | 1 | UC-09 | remind me to drink water in five minutes | `null` |
| UC09-REJ-13-N1 | 2 | UC-09 | hey assistant, remind me to drink water in five minutes | `null` |
| UC09-REJ-13-N2 | 2 | UC-09 | remind me to drink water in five minutes, please | `null` |
| UC09-REJ-14 | 1 | UC-09 | remind me to take medicine in ten minutes | `null` |
| UC09-REJ-14-N1 | 2 | UC-09 | hey assistant, remind me to take medicine in ten minutes | `null` |
| UC09-REJ-14-N2 | 2 | UC-09 | remind me to take medicine in ten minutes, please | `null` |
| UC09-REJ-15 | 1 | UC-09 | remind me to check the oven in ten minutes | `null` |
| UC09-REJ-15-N1 | 2 | UC-09 | hey assistant, remind me to check the oven in ten minutes | `null` |
| UC09-REJ-15-N2 | 2 | UC-09 | remind me to check the oven in ten minutes, please | `null` |
| UC09-REJ-16 | 1 | UC-09 | remind me to stretch tomorrow | `null` |
| UC09-REJ-16-N1 | 2 | UC-09 | hey assistant, remind me to stretch tomorrow | `null` |
| UC09-REJ-16-N2 | 2 | UC-09 | remind me to stretch tomorrow, please | `null` |
| UC09-REJ-17 | 1 | UC-09 | add butter to my shopping list | `null` |
| UC09-REJ-17-N1 | 2 | UC-09 | hey assistant, add butter to my shopping list | `null` |
| UC09-REJ-17-N2 | 2 | UC-09 | add butter to my shopping list, please | `null` |
| UC09-REJ-18 | 1 | UC-09 | add milk to my work list | `null` |
| UC09-REJ-18-N1 | 2 | UC-09 | hey assistant, add milk to my work list | `null` |
| UC09-REJ-18-N2 | 2 | UC-09 | add milk to my work list, please | `null` |
| UC09-REJ-19 | 1 | UC-09 | delete my shopping list | `null` |
| UC09-REJ-19-N1 | 2 | UC-09 | hey assistant, delete my shopping list | `null` |
| UC09-REJ-19-N2 | 2 | UC-09 | delete my shopping list, please | `null` |
| UC09-REJ-20 | 1 | UC-09 | remind me to call jamie in ten minutes | `null` |
| UC09-REJ-20-N1 | 2 | UC-09 | hey assistant, remind me to call jamie in ten minutes | `null` |
| UC09-REJ-20-N2 | 2 | UC-09 | remind me to call jamie in ten minutes, please | `null` |
| UC10-REJ-01 | 1 | UC-10 | do not call alex kim | `null` |
| UC10-REJ-01-N1 | 2 | UC-10 | hey assistant, do not call alex kim | `null` |
| UC10-REJ-01-N2 | 2 | UC-10 | do not call alex kim, please | `null` |
| UC10-REJ-02 | 1 | UC-10 | do not message jamie cruz i will be late | `null` |
| UC10-REJ-02-N1 | 2 | UC-10 | hey assistant, do not message jamie cruz i will be late | `null` |
| UC10-REJ-02-N2 | 2 | UC-10 | do not message jamie cruz i will be late, please | `null` |
| UC10-REJ-03 | 1 | UC-10 | call alex kim and call jamie cruz | `null` |
| UC10-REJ-03-N1 | 2 | UC-10 | hey assistant, call alex kim and call jamie cruz | `null` |
| UC10-REJ-03-N2 | 2 | UC-10 | call alex kim and call jamie cruz, please | `null` |
| UC10-REJ-04 | 1 | UC-10 | message alex kim i am on my way and pause music | `null` |
| UC10-REJ-04-N1 | 2 | UC-10 | hey assistant, message alex kim i am on my way and pause music | `null` |
| UC10-REJ-04-N2 | 2 | UC-10 | message alex kim i am on my way and pause music, please | `null` |
| UC10-REJ-05 | 1 | UC-10 | call | `null` |
| UC10-REJ-05-N1 | 2 | UC-10 | hey assistant, call | `null` |
| UC10-REJ-05-N2 | 2 | UC-10 | call, please | `null` |
| UC10-REJ-06 | 1 | UC-10 | message alex kim | `null` |
| UC10-REJ-06-N1 | 2 | UC-10 | hey assistant, message alex kim | `null` |
| UC10-REJ-06-N2 | 2 | UC-10 | message alex kim, please | `null` |
| UC10-REJ-07 | 1 | UC-10 | tell jamie cruz | `null` |
| UC10-REJ-07-N1 | 2 | UC-10 | hey assistant, tell jamie cruz | `null` |
| UC10-REJ-07-N2 | 2 | UC-10 | tell jamie cruz, please | `null` |
| UC10-REJ-08 | 1 | UC-10 | call alex kim jamie cruz | `null` |
| UC10-REJ-08-N1 | 2 | UC-10 | hey assistant, call alex kim jamie cruz | `null` |
| UC10-REJ-08-N2 | 2 | UC-10 | call alex kim jamie cruz, please | `null` |
| UC10-REJ-09 | 1 | UC-10 | message alex kim i am on my way i will be late | `null` |
| UC10-REJ-09-N1 | 2 | UC-10 | hey assistant, message alex kim i am on my way i will be late | `null` |
| UC10-REJ-09-N2 | 2 | UC-10 | message alex kim i am on my way i will be late, please | `null` |
| UC10-REJ-10 | 1 | UC-10 | call someone | `null` |
| UC10-REJ-10-N1 | 2 | UC-10 | hey assistant, call someone | `null` |
| UC10-REJ-10-N2 | 2 | UC-10 | call someone, please | `null` |
| UC10-REJ-11 | 1 | UC-10 | message someone please call me | `null` |
| UC10-REJ-11-N1 | 2 | UC-10 | hey assistant, message someone please call me | `null` |
| UC10-REJ-11-N2 | 2 | UC-10 | message someone please call me, please | `null` |
| UC10-REJ-12 | 1 | UC-10 | call alex kim not jamie cruz | `null` |
| UC10-REJ-12-N1 | 2 | UC-10 | hey assistant, call alex kim not jamie cruz | `null` |
| UC10-REJ-12-N2 | 2 | UC-10 | call alex kim not jamie cruz, please | `null` |
| UC10-REJ-13 | 1 | UC-10 | call alex king | `null` |
| UC10-REJ-13-N1 | 2 | UC-10 | hey assistant, call alex king | `null` |
| UC10-REJ-13-N2 | 2 | UC-10 | call alex king, please | `null` |
| UC10-REJ-14 | 1 | UC-10 | call jamie cross | `null` |
| UC10-REJ-14-N1 | 2 | UC-10 | hey assistant, call jamie cross | `null` |
| UC10-REJ-14-N2 | 2 | UC-10 | call jamie cross, please | `null` |
| UC10-REJ-15 | 1 | UC-10 | call morgan dean | `null` |
| UC10-REJ-15-N1 | 2 | UC-10 | hey assistant, call morgan dean | `null` |
| UC10-REJ-15-N2 | 2 | UC-10 | call morgan dean, please | `null` |
| UC10-REJ-16 | 1 | UC-10 | call taylor smith | `null` |
| UC10-REJ-16-N1 | 2 | UC-10 | hey assistant, call taylor smith | `null` |
| UC10-REJ-16-N2 | 2 | UC-10 | call taylor smith, please | `null` |
| UC10-REJ-17 | 1 | UC-10 | message alex kim i am on the way | `null` |
| UC10-REJ-17-N1 | 2 | UC-10 | hey assistant, message alex kim i am on the way | `null` |
| UC10-REJ-17-N2 | 2 | UC-10 | message alex kim i am on the way, please | `null` |
| UC10-REJ-18 | 1 | UC-10 | message alex kim i will be early | `null` |
| UC10-REJ-18-N1 | 2 | UC-10 | hey assistant, message alex kim i will be early | `null` |
| UC10-REJ-18-N2 | 2 | UC-10 | message alex kim i will be early, please | `null` |
| UC10-REJ-19 | 1 | UC-10 | tell jamie cruz please call us | `null` |
| UC10-REJ-19-N1 | 2 | UC-10 | hey assistant, tell jamie cruz please call us | `null` |
| UC10-REJ-19-N2 | 2 | UC-10 | tell jamie cruz please call us, please | `null` |
| UC10-REJ-20 | 1 | UC-10 | message morgan lee i am home | `null` |
| UC10-REJ-20-N1 | 2 | UC-10 | hey assistant, message morgan lee i am home | `null` |
| UC10-REJ-20-N2 | 2 | UC-10 | message morgan lee i am home, please | `null` |

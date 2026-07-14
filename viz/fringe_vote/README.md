# Fringe "Genre Roulette" — Audience Interaction System

Three web pages + a tiny robot bridge that let the audience steer the show:

1. **Genre roulette** (`roulette.html`) — projected wheel, tap to land on one of 4 improv genres.
2. **Live vote** (`vote.html`) — phones scan a QR and continuously vote **mirror** vs **contrast**.
3. **Results backdrop** (`results.html`) — full-screen two-tone gradient driven by the live vote.

The rolling majority automatically switches the robot between the **mirror** model (`model_id 1`)
and the **contrast** model (`model_id 2`) via the existing `/vam/switch_model` ROS2 service — with
hysteresis so it never thrashes, and a manual-override freeze.

```
phones ─HTTPS─▶ Hostinger (PHP + MariaDB)            robot laptop
  vote.html      api/vote.php   (record vote)        ┌──────────────────────────────┐
laptop browser   api/tally.php  (tally + ratio) ◀────┤ vote_robot_bridge (viz Docker)│
  roulette.html  api/genre.php  (roulette pick)  GET │  ~1/s GET tally.php           │
  results.html ◀─poll─────────────────────────  1/s  │  debounce ▶ /vam/switch_model │
                                                      └──────────────────────────────┘
```

Why this split: shared hosting absorbs all 50–100 phones; the robot laptop only makes one tiny GET
per second. No WebSockets, no tunnel, no inbound exposure of the laptop.

---

## Part A — Hostinger (once, no robot needed)

Your Premium plan provides PHP 8.x + MariaDB (MySQL-compatible) + phpMyAdmin — everything here.

1. **Where it lives — a folder, no subdomain needed.** Your site is WordPress; drop a plain folder
   beside it, e.g. `public_html/fringe/`, served at `https://yourdomain.com/fringe/vote.html`.
   WordPress's `.htaccess` only rewrites URLs to itself when the file/folder doesn't exist
   (`!-f`/`!-d`), so a real folder with real files bypasses WordPress. Cautions: (a) don't reuse a
   WordPress page slug as the folder name; (b) these are static files + PHP, **not** a WP page.

2. **Create a database.** hPanel → **Databases → MySQL Databases** → *Create*. Note the four values:
   database **name** (`uXXXXXX_fringe`), **username** (`uXXXXXX_voteuser`), **password**, **host**
   (`localhost`).

3. **Create the tables.** Click **Enter phpMyAdmin** → **SQL** tab → paste all of `web/schema.sql`
   → **Go**. The `votes` and `genre` tables appear on the left.

4. **Upload the site.** hPanel → **Files → File Manager** → into your folder (e.g.
   `…/public_html/fringe`) → upload the **contents of `web/`** (the `.html`, `vote-api.js`, and the
   `api/` folder). Dragging a zip and using *Extract* is easiest.

5. **Add DB credentials.** In File Manager, copy `config.example.php` → `config.php`, *Edit*, fill in
   the four values from step 2, Save. (Typos here are the #1 cause of "won't connect".)

6. **HTTPS.** Hostinger auto-issues SSL; if the page says "not secure", **Security → SSL** → enable.

7. **Test.** Open `https://yourdomain.com/fringe/api/tally.php` — it should print JSON (not an
   error). Then `vote.html` on a phone, `results.html` on a laptop; watch the gradient move as you
   tap. Quick check from any terminal:
   ```bash
   curl -s -X POST https://yourdomain.com/fringe/api/vote.php \
        -H 'Content-Type: application/json' -d '{"clientId":"test1","choice":"contrast"}'
   curl -s "https://yourdomain.com/fringe/api/tally.php"
   ```

8. **QR code.** Generate a QR pointing at `https://yourdomain.com/fringe/vote.html`; print/show it.

---

## Part B — Robot laptop (auto-switch bridge)

The bridge lives in the existing `vam_viz_bridge` package and runs in the **viz Docker**. It needs
the `vam_interfaces` service type, which the inference (`rapp_vam`) container builds into the shared
`ros2_ws/install/` — the viz entrypoint sources that overlay automatically if present.

1. `git pull` on the robot laptop.
2. Make sure the **inference container has been built at least once** (it produces `vam_interfaces`):
   that's part of your normal robot startup.
3. Rebuild/start the viz container (picks up the new node + entrypoint change):
   ```bash
   cd docker && docker compose -f docker-compose.viz.yml up -d --build
   ```
4. Launch the bridge (separate from the streaming bridge, same container):
   ```bash
   docker exec -it rapp_viz bash
   ros2 launch vam_viz_bridge vote_bridge.launch.py \
        state_url:=https://yourdomain.com/fringe/api/tally.php
   ```
   The log prints the live tally each second and `SWITCH -> contrast (model_id 2)` on a committed
   change.

### Manual override (take control instantly)
```bash
ros2 param set /vote_robot_bridge auto_switch false   # freeze (you drive manually)
ros2 param set /vote_robot_bridge auto_switch true    # resume auto
# or, programmatically:
ros2 topic pub -1 /vote/auto_enable std_msgs/msg/Bool "{data: false}"
```
Your existing manual fallback always works: `ros2 service call /vam/switch_model
vam_interfaces/srv/SwitchModel "{model_id: 1}"`.

### Tunables (launch args or `ros2 param set`)
| param | default | meaning |
|-------|---------|---------|
| `state_url` | localhost | full URL of `tally.php` |
| `active_window` | 25 | seconds a vote counts as "active" (matches phone heartbeat) |
| `enter_contrast` / `enter_mirror` | 0.55 / 0.45 | deadband: ratio thresholds to want each mode |
| `hold_seconds` | 3.0 | majority must persist this long before switching |
| `min_switch_interval` | 12.0 | cooldown between switches (a model swap resets the pipeline) |
| `mirror_model_id` / `contrast_model_id` | 1 / 2 | from `vam_models.yaml` |
| `state_token` | "" | set if you enabled `state_token` in `config.php` |

`ratio` from `tally.php`: **0 = all mirror, 1 = all contrast, 0.5 = no votes** (neutral → no switch).

---

## Part C — Show-day checklist (~5 min)
- Robot stack up → inference container running (hosts `/vam/switch_model`).
- `vote_robot_bridge` launched; log shows live counts.
- `results.html` full-screen on the backdrop machine; `roulette.html` ready on the projector.
- QR visible; cast one test vote from your phone and watch it register + the robot react.
- Keep a terminal with the `auto_switch false` command ready in case you want manual control.

---

## Visual redesign (Claude Design)
The three pages are plain but fully working. To restyle them, the design only needs to keep the
`vote-api.js` calls and these hook IDs:
- `vote.html`: buttons `#btn-mirror`, `#btn-contrast`; tally `#tally`; call `VoteAPI.sendVote(...)`.
- `roulette.html`: `#spin`, `#result`; call `VoteAPI.pickGenre()` for the landing genre.
- `results.html`: `VoteAPI.onState(cb)` / `VoteAPI.getRatio()` (0=mirror … 1=contrast) to drive the
  gradient luminance.

## Files
```
viz/fringe_vote/web/        -> upload to Hostinger
  schema.sql            -> paste into phpMyAdmin SQL tab
  config.example.php    -> copy to config.php, fill DB creds (git-ignored)
  vote-api.js           -> the JS contract
  vote.html  roulette.html  results.html
  api/db.php api/vote.php api/tally.php api/genre.php

ros2_ws/src/vam_viz_bridge/
  vam_viz_bridge/vote_robot_bridge.py   -> the bridge node
  launch/vote_bridge.launch.py
```
Existing inference/hardware/streaming code is untouched; the only edits are additive: one
`setup.py` entry-point line, a `vam_interfaces` dep in `package.xml`, and a guarded overlay-source
in the viz `entrypoint.sh`.

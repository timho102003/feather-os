# Messaging Integrations

You can talk to your Feather agent from your phone instead of your
terminal. Three platforms are supported: Telegram, LINE, and WhatsApp.

When an integration is connected, every message you send to the bot
goes into the same chat session you use in the TUI. The agent's
replies show up in the platform of your choice and in your terminal
transcript.

## Pick a platform based on what you want

* **Telegram** is the easiest. No public URL needed. Bot token from
  BotFather, paste, done.
* **LINE** and **WhatsApp** use webhooks. You need a public URL that
  your phone-side platform can POST to. Most people use a tunnel like
  `ngrok` or `cloudflared`.

## Telegram

### What you need

A Telegram account, the [BotFather](https://t.me/BotFather) bot, and
an API token. To get one:

1. Open Telegram, search for `@BotFather`.
2. Send `/newbot`. Pick a name and a username.
3. BotFather replies with a token that looks like
   `123456789:ABCdef...`. Save it somewhere private.

### Connect

In a chat:

```
/telegram connect 123456789:ABCdef...
```

That's it. Feather:

1. Validates the token by calling `getMe`.
2. Clears any old webhook the bot might have.
3. Starts long-polling Telegram for new messages.

You will see a confirmation. Now message the bot from your phone.
Whatever you send shows up in your terminal session and the agent's
reply goes back over Telegram.

### Disconnect

```
/telegram disconnect
```

The bot stops receiving messages until you connect it again.

### Status

```
/telegram status
```

Shows whether the adapter is connected and the bot's username.

## LINE

### What you need

* A LINE developer account at <https://developers.line.biz/>.
* A Messaging API channel. Take note of the **channel secret** and
  **channel access token** (long-lived).
* A public URL that LINE can reach. Local development: install
  [ngrok](https://ngrok.com/) or
  [cloudflared](https://github.com/cloudflare/cloudflared) and tunnel
  port 8765 (Feather's default webhook port).

### Connect

Run a tunnel first:

```bash
ngrok http 8765
```

It will print a public HTTPS URL like
`https://abc123.ngrok-free.app`. Configure that URL plus
`/messaging/webhook/line` as the Webhook URL in your LINE channel
settings (so the full webhook is `https://abc123.ngrok-free.app/messaging/webhook/line`).
Enable "Use webhook" and disable "Auto-reply" in the LINE console.

Then in Feather:

```
/line connect <channel_secret> <channel_access_token>
```

Send a message to your LINE official account. The conversation flows
into your Feather session.

### Disconnect

```
/line disconnect
```

### Status

```
/line status
```

## WhatsApp

### What you need

* A Meta Developer account at <https://developers.facebook.com/>.
* A WhatsApp Business Cloud API app with a registered phone number.
  You'll need:
  * **Phone Number ID**: found under the WhatsApp app dashboard.
  * **Access Token**: long-lived system user token from the App's
    settings.
  * **Verify Token**: any string you make up; you'll paste the same
    string into the Meta webhook config and the connect command.
  * **App Secret**: used to sign webhook payloads.
* A public URL that Meta can reach. Same tunneling story as LINE.

### Connect

Run a tunnel:

```bash
ngrok http 8765
```

Set the Meta webhook URL to
`https://<your-tunnel>/messaging/webhook/whatsapp` and the verify
token to whatever you'll pass in the connect command.

Then:

```
/whatsapp connect <phone_id> <access_token> <verify_token> <app_secret>
```

Meta will fire a verification GET against the webhook; Feather
responds. Once Meta confirms, your number is live.

Send a message to the WhatsApp Business number. The conversation
flows into your Feather session.

### Disconnect

```
/whatsapp disconnect
```

### Status

```
/whatsapp status
```

## See everything at once

```
/integrations
```

Shows the connected state for all three platforms in one panel.

## How it works under the hood

* The Telegram adapter uses long polling, so no public URL is needed.
  It opens a connection to Telegram and waits for updates.
* LINE and WhatsApp adapters share a single aiohttp webhook server
  bound to `127.0.0.1:8765`. The server only starts when at least one
  webhook is registered, and it stops when the last one disconnects.
* Both webhook adapters use the same chat session as your terminal,
  so the conversation context is preserved.
* Messages from your phone are routed through the **lead** agent. The
  agent's reply goes back to the platform you sent from.

The bind is loopback by default. The tunnel is what makes the webhook
reachable from the public internet. **Don't bind to 0.0.0.0 unless you
know what you're doing**, or you'll expose the webhook directly.

## Things to avoid

* **Don't paste your Telegram token in a public chat.** It is the only
  credential the bot needs.
* **Don't expose the webhook server to the public internet without a
  tunnel.** The server has rudimentary signature checking but is not
  designed to be the only line of defense.
* **Don't connect the same bot to two Feather instances.** They will
  fight over messages.

## Things that don't work yet

* No multi-user support. The integration is one phone number to one
  Feather user.
* No file/image upload from the phone yet (text only).
* No group-chat support; the bot replies in 1:1 chats.

## Next

* Want scheduled messages from the bot? See
  [scheduling.md](scheduling.md). Set up a cron that fires a prompt
  while the integration is connected, and the reply goes to your phone.
* Want to debug a webhook? Look at the runtime log:
  `~/.feather/state/logs/feather.log` (global mode) or
  `./.feather/logs/feather.log` (project mode).

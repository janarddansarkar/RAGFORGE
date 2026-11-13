# Running RAGForge on a VPS (no Docker)

A plain `git clone` and one command. This was written for an Ubuntu/Debian server; the commands are examples. The
app itself was tested locally (a real server and a real browser), not on a VPS.

## What you need
- **RAM:** about 2 GB is comfortable. The app uses about 560 MB for one user after loading everything; the quality
  scoring library adds about 160 MB the first time it is used. 1 GB works only with swap.
- **Disk:** about 1 GB for the Python packages. Uploads are not kept.
- **Packages:** `git`, `curl`, and `tesseract-ocr` (needed only to read scanned PDFs).
- **A domain name** pointing at the server, so the page can use HTTPS. Users type their API keys into the page, so do
  not serve it over plain HTTP.

## 1. Install and run

```bash
sudo apt update && sudo apt install -y git curl tesseract-ocr
curl -LsSf https://astral.sh/uv/install.sh | sh        # installs uv; open a new shell afterwards

git clone -b ragforge_api https://github.com/janarddansarkar/RAGFORGE.git
cd RAGFORGE
uv run streamlit run src/app.py --server.address 127.0.0.1 --server.port 8501
```

The first run installs the Python packages (and Python 3.12 if the server has another version). The app now listens
only on the server itself (127.0.0.1), so put a web server in front of it (step 3). `.streamlit/config.toml` already
sets headless mode, a 25 MB cap per uploaded file and no usage statistics.

## 2. Keep it running (systemd)

Create `/etc/systemd/system/ragforge.service` (change `User` and the paths to yours):

```ini
[Unit]
Description=RAGForge
After=network.target

[Service]
User=ragforge
WorkingDirectory=/home/ragforge/RAGFORGE
ExecStart=/home/ragforge/.local/bin/uv run streamlit run src/app.py --server.address 127.0.0.1 --server.port 8501
Restart=always
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ragforge
sudo systemctl status ragforge
```

`PrivateTmp=true` gives the service its own temp folder, which systemd removes when the service stops: a second guarantee
that no upload survives a restart.

## 3. HTTPS with Caddy

```bash
sudo apt install -y caddy
```

`/etc/caddy/Caddyfile`:

```
your-domain.example {
    reverse_proxy 127.0.0.1:8501
}
```

```bash
sudo systemctl reload caddy
sudo ufw allow OpenSSH && sudo ufw allow 80,443/tcp && sudo ufw enable   # keep 8501 closed to the outside
```

Caddy gets the certificate by itself and passes the WebSocket Streamlit needs. With nginx you must add the WebSocket
headers (`Upgrade` and `Connection`) yourself.

## 4. Update

```bash
cd ~/RAGFORGE && git pull && sudo systemctl restart ragforge
```

## What the server keeps
- **Users' PDFs:** deleted as soon as they are indexed.
- **Their index and answer cache:** in a private folder per browser session under the system temp folder
  (`ragforge_workspaces`), deleted when the documents are replaced, when the session ends (a reload, a closed tab, a new
  session), and on every server restart.
- **API keys:** only in the session's memory; never written to disk or logged.
- **Limits:** at most 20 pages in total per user (`max_pages` in `config.toml`) and 25 MB per file.

## Things to know
- **No login or rate limit.** The model costs are paid with each user's own key, but anyone who can reach the page can
  use your server's CPU and memory. Put it behind a login (Caddy `basic_auth`, a VPN or an IP allow-list) if it is not
  meant to be public.
- If a user's connection drops and another session starts during the gap, their private folder can be cleaned up. The
  app then tells them to upload again.
- Scanned PDFs are slow on a small server (OCR runs on the CPU).

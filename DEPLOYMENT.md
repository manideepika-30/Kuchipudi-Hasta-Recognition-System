# Production Deployment Guide — Nritya AI Platform

This document outlines the standard deployment architecture and procedures for hosting **Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System** in a production Linux / Cloud environment.

---

## 1. System Requirements & Recommendations

| Component | Minimum | Recommended |
|---|---|---|
| **CPU** | 2 Cores (x86_64 or ARM64) | 4+ Cores (2.4 GHz+) |
| **RAM** | 4 GB | 8 GB |
| **Storage** | 20 GB SSD | 50 GB NVMe SSD |
| **OS** | Ubuntu 22.04 LTS / Debian 12 | Ubuntu 24.04 LTS |
| **Network** | 100 Mbps | 1 Gbps |
| **SSL/TLS** | Required (HTTPS is mandatory for WebRTC camera access) | Valid TLS certificate via Let's Encrypt |

> [!IMPORTANT]
> **HTTPS Requirement for WebRTC**: Modern web browsers strictly require a secure origin (`https://` or `localhost`) to grant webcam access for the Practice Studio (`navigator.mediaDevices.getUserMedia`). Ensure a valid SSL/TLS certificate is configured on your production reverse proxy.

---

## 2. Production Environment Configuration

Create a dedicated `.env` file inside `backend/` with production-hardened credentials:

```ini
# backend/.env
ENVIRONMENT=production
DEBUG=False

# SQLite Database Connection (or PostgreSQL if scaled)
DATABASE_URL=sqlite:///./kuchipudi.db

# Cryptographically Secure Secret Key (generate via: openssl rand -hex 32)
SECRET_KEY=9a4c8e71b2d04f32e91a56c4d7b8901234ef567890abcdef1234567890abcdef
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Production Allowed Origins
CORS_ORIGINS=https://kuchipudi.org,https://www.kuchipudi.org

# AI Engine Configuration
AI_WEIGHTS_PATH=deployment_clean/nritya_ai_weights.keras
AI_DEVICE=cpu
CONFIDENCE_THRESHOLD=0.70
```

---

## 3. Frontend Production Build

The React application uses Vite to build minified, hash-versioned static bundles:

```bash
cd frontend
npm install --production=false
npm run build
```

This compiles optimized assets to `frontend/dist/`:
- `dist/index.html` (~1.5 kB)
- `dist/assets/index-*.js` (~312 kB, gzip ~79 kB)
- `dist/assets/index-*.css` (~5 kB, gzip ~1.5 kB)

---

## 4. Nginx Reverse Proxy & SSL Configuration

Configure Nginx to serve the frontend static files directly and reverse proxy API calls to the FastAPI backend:

```nginx
# /etc/nginx/sites-available/kuchipudi.conf
server {
    listen 80;
    server_name kuchipudi.org www.kuchipudi.org;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name kuchipudi.org www.kuchipudi.org;

    # SSL Certificates (managed via Certbot)
    ssl_certificate /etc/letsencrypt/live/kuchipudi.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/kuchipudi.org/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Frontend Static Files
    root /var/www/kuchipudi/frontend/dist;
    index index.html;

    # Single Page App routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API Proxy to FastAPI Backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 90;
    }

    # Static Assets Cache
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 30d;
        add_header Cache-Control "public, no-transform";
    }

    client_max_body_size 10M;
}
```

Enable site and reload:
```bash
sudo ln -s /etc/nginx/sites-available/kuchipudi.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 5. Backend Daemon via Systemd

Create a persistent background service managed by `systemd`:

```ini
# /etc/systemd/system/kuchipudi-backend.service
[Unit]
Description=Nritya AI Kuchipudi Backend Service
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/kuchipudi/backend
Environment="PATH=/var/www/kuchipudi/backend/.venv/bin"
ExecStart=/var/www/kuchipudi/backend/.venv/bin/uvicorn app.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 2 \
    --timeout-keep-alive 30

Restart=always
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable kuchipudi-backend
sudo systemctl start kuchipudi-backend
sudo systemctl status kuchipudi-backend
```

---

## 6. Database Maintenance & Backups

For SQLite deployments, perform automated daily backups using SQLite online backup:

```bash
#!/bin/bash
# /var/www/kuchipudi/scripts/backup_db.sh
BACKUP_DIR="/var/backups/kuchipudi"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$BACKUP_DIR"

sqlite3 /var/www/kuchipudi/backend/kuchipudi.db ".backup '$BACKUP_DIR/kuchipudi_$TIMESTAMP.db'"
# Keep last 14 days
find "$BACKUP_DIR" -type f -name "*.db" -mtime +14 -delete
```

Add to cron:
```bash
crontab -e
# Run daily at 02:00 AM
0 2 * * * /bin/bash /var/www/kuchipudi/scripts/backup_db.sh
```

---

## 7. Post-Deployment Verification Checklist

- [ ] HTTPS is active and browser displays secure padlock icon.
- [ ] Practice Studio requests and successfully acquires webcam feed.
- [ ] MediaPipe hand landmarks render properly on the canvas overlay.
- [ ] API status endpoint returns `200 OK` at `https://your-domain.com/api/ai/status`.
- [ ] Real-time classification responds with authentic gesture name and confidence.
- [ ] Student registration, login, logout, and token invalidation are functioning.
- [ ] Admin panel access is restricted to authenticated administrator accounts.
- [ ] All 7 modules and 92 lessons load with correct shlokas and images.

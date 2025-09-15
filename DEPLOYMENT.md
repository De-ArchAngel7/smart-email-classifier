# 🚀 Smart Email Classifier - Deployment Guide

## 📋 **Platform Overview**

| Service | Purpose | URL | Status |
|---------|---------|-----|--------|
| **Railway** | Backend API | `https://smart-email-classifier-production.up.railway.app` | Ready to deploy |
| **Vercel** | Frontend App | `https://smart-email-classifier.vercel.app` | Ready to deploy |
| **MongoDB Atlas** | Database | `mongodb+srv://...` | Optional |

---

## 1️⃣ **Backend Deployment (Railway)**

### **Step 1: Create Railway Account**
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Connect your GitHub account

### **Step 2: Deploy Backend**
1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose `ElBalor/smart-email-classifier` or `De-ArchAngel7/smart-email-classifier`
4. Railway will auto-detect Python and install dependencies
5. Set environment variables (if needed):
   - `PORT`: `8000` (auto-set by Railway)
   - `PYTHON_VERSION`: `3.10.12`

### **Step 3: Get Backend URL**
- Railway will provide a URL like: `https://smart-email-classifier-production.up.railway.app`
- Copy this URL for frontend configuration

---

## 2️⃣ **Frontend Deployment (Vercel)**

### **Step 1: Create Vercel Account**
1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub
3. Connect your GitHub account

### **Step 2: Deploy Frontend**
1. Click **"New Project"**
2. Import `ElBalor/smart-email-classifier` or `De-ArchAngel7/smart-email-classifier`
3. Set **Root Directory** to `frontend`
4. Vercel will auto-detect Next.js and build
5. Add environment variable:
   - `NEXT_PUBLIC_API_URL`: `https://your-railway-url.up.railway.app`

### **Step 3: Update Frontend API URL**
After getting your Railway URL, update the frontend:

```typescript
// In frontend/src/app/page.tsx
const response = await axios.post(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/classify`, {
  text: emailText,
});
```

---

## 3️⃣ **Database Setup (Optional - MongoDB Atlas)**

### **Step 1: Create MongoDB Atlas Account**
1. Go to [mongodb.com/atlas](https://mongodb.com/atlas)
2. Create free account
3. Create a new cluster

### **Step 2: Get Connection String**
1. Click **"Connect"** on your cluster
2. Choose **"Connect your application"**
3. Copy the connection string
4. Replace `<password>` with your database password

### **Step 3: Add to Railway Environment Variables**
- `MONGODB_URI`: `mongodb+srv://username:password@cluster.mongodb.net/smart-email-classifier`

---

## 4️⃣ **CI/CD Setup**

### **Automatic Deployments**
Both Railway and Vercel will automatically deploy when you push to the `main` branch.

### **Manual Deployment**
Use the mirror scripts:
```bash
# Windows
.\mirror.bat

# Linux/Mac
./mirror.sh
```

---

## 5️⃣ **Testing Production**

### **Backend Test**
```bash
curl -X POST "https://your-railway-url.up.railway.app/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "I need help with my billing issue"}'
```

### **Frontend Test**
1. Visit your Vercel URL
2. Paste email text
3. Click "Classify"
4. Verify results

---

## 6️⃣ **Environment Variables Summary**

### **Railway (Backend)**
```
PORT=8000
PYTHON_VERSION=3.10.12
MONGODB_URI=mongodb+srv://... (optional)
```

### **Vercel (Frontend)**
```
NEXT_PUBLIC_API_URL=https://your-railway-url.up.railway.app
```

---

## 7️⃣ **Monitoring & Maintenance**

### **Railway Dashboard**
- Monitor backend performance
- View logs
- Manage environment variables

### **Vercel Dashboard**
- Monitor frontend performance
- View build logs
- Manage domains

### **MongoDB Atlas Dashboard**
- Monitor database usage
- View query performance
- Manage collections

---

## 🎯 **Quick Start Commands**

```bash
# 1. Deploy backend to Railway
# - Go to railway.app
# - Connect GitHub repo
# - Deploy automatically

# 2. Deploy frontend to Vercel
# - Go to vercel.com
# - Import GitHub repo
# - Set root directory to 'frontend'
# - Add NEXT_PUBLIC_API_URL environment variable

# 3. Test deployment
curl -X POST "https://your-railway-url.up.railway.app/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Test email"}'
```

---

## 🚨 **Troubleshooting**

### **Common Issues**
1. **CORS Error**: Update CORS origins in `main.py`
2. **Build Failed**: Check Python version and dependencies
3. **API Not Found**: Verify environment variables
4. **Model Loading**: Check Railway logs for model download issues

### **Support**
- Railway: [docs.railway.app](https://docs.railway.app)
- Vercel: [vercel.com/docs](https://vercel.com/docs)
- MongoDB: [docs.mongodb.com](https://docs.mongodb.com)

---

## 🎉 **Success!**

Once deployed, your Smart Email Classifier will be live at:
- **Frontend**: `https://smart-email-classifier.vercel.app`
- **Backend**: `https://smart-email-classifier-production.up.railway.app`

The app will automatically classify emails into categories like Billing, Technical Issue, Refund, Spam, etc. with confidence scores!

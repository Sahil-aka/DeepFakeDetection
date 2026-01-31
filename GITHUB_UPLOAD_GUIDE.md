# GitHub Upload Guide for DeepFake Detection Project

## Quick Upload (Recommended)

### Option 1: Using the Batch Script
Simply double-click `upload_to_github.bat` in your project folder. This will:
1. Add all your files to git
2. Commit the changes
3. Push to GitHub

### Option 2: Manual Commands
Open Command Prompt in your project folder and run:

```bash
git add .
git commit -m "Update project for GitHub showcase"
git push origin main
```

If `main` doesn't work, try:
```bash
git push origin master
```

---

## First Time Setup (If Repository Doesn't Exist)

### Step 1: Create GitHub Repository
1. Go to https://github.com
2. Click the "+" icon → "New repository"
3. Name it: `DeepFakeDetection` (or any name you prefer)
4. **Don't** initialize with README (you already have one)
5. Click "Create repository"

### Step 2: Connect Your Local Project
```bash
git remote add origin https://github.com/YOUR_USERNAME/DeepFakeDetection.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## If You Get Errors

### Error: "Repository not found"
- Make sure you've created the repository on GitHub
- Check that the remote URL is correct: `git remote -v`
- Update the URL if needed: `git remote set-url origin https://github.com/YOUR_USERNAME/DeepFakeDetection.git`

### Error: "Large files detected"
Your model file `deepfake_cnn_gpu.h5` is 128MB. GitHub has a 100MB limit. Options:

**Option A: Use Git LFS (Recommended)**
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push origin main
```

**Option B: Remove model from git**
Add to `.gitignore`:
```
*.h5
*.keras
```

Then:
```bash
git rm --cached deepfake_cnn_gpu.h5
git commit -m "Remove large model file"
git push origin main
```

Users can download the model separately or train their own.

### Error: "Authentication failed"
- Use a Personal Access Token instead of password
- Go to GitHub Settings → Developer settings → Personal access tokens
- Generate a new token with `repo` permissions
- Use the token as your password when pushing

---

## What Gets Uploaded

✅ **Included:**
- Source code (`app.py`, `src/` folder)
- Frontend files (`frontend/` folder)
- Documentation (`README.md`, `LICENSE`)
- Configuration files (`requirements.txt`, `.gitignore`)

❌ **Excluded (in .gitignore):**
- `__pycache__/` - Python cache files
- `.venv/` - Virtual environment
- `.idea/` - IDE settings
- `data/Dataset/` - Training dataset (too large)
- `*.h5` files - Model files (if uncommented in .gitignore)

---

## Making Your Project Look Professional

### 1. Add a Great README
Your README is already good! Consider adding:
- Screenshots of the web interface
- Demo GIF showing the upload and prediction
- Badges (build status, license, etc.)

### 2. Add Topics/Tags
On GitHub, add topics like:
- `deepfake-detection`
- `machine-learning`
- `cnn`
- `fastapi`
- `computer-vision`

### 3. Create a Demo
Consider deploying to:
- **Hugging Face Spaces** (Free, easy for ML projects)
- **Railway** (Free tier available)
- **Render** (Free tier available)

### 4. Add Screenshots
Create a `screenshots/` folder and add images of your UI

---

## Troubleshooting

### Check Current Status
```bash
git status
```

### Check Remote Connection
```bash
git remote -v
```

### View Commit History
```bash
git log --oneline
```

### Force Push (Use Carefully!)
```bash
git push -f origin main
```

---

## Need Help?

If you encounter any issues:
1. Check the error message carefully
2. Make sure you're logged into GitHub
3. Verify your repository exists
4. Check your internet connection

Good luck with your GitHub showcase! 🚀

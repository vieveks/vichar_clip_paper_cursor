# Kickstart Guide for Cursor AI Sessions

This guide helps maintain consistency, proper documentation, and organized development across all Cursor chat sessions.

## 📋 Core Principles

1. **Always log progress** in `updates.md` with date-wise entries
2. **Commit changes properly** to git with meaningful messages
3. **Progress incrementally** - avoid creating unnecessary files
4. **Maintain documentation** - update relevant docs when making changes
5. **Test before committing** - ensure code works before pushing

---

## 📝 Daily Progress Logging

### Using `updates.md`

**ALWAYS update `updates.md` at the end of each session or when completing significant work.**

Format:
```markdown
## YYYY-MM-DD

### Feature/Change Name
- Description of what was done
- Files created/modified
- Important notes or decisions
```

**Example:**
```markdown
## 2025-11-22

### Hugging Face Dataset Integration
- Created dataset download script (`utils/download_hf_chess_dataset.py`)
- Implemented dataset loader with native/custom split support
- Added training script with proper train/val/test splits
- Created comprehensive documentation in `docs/HF_DATASET_METHODOLOGY.md`
- All changes committed to git with proper messages
```

### When to Update

- ✅ After completing a feature or task
- ✅ After fixing bugs
- ✅ After adding new files or scripts
- ✅ After updating documentation
- ✅ At the end of each working session

---

## 🔄 Git Workflow

### Commit Best Practices

1. **Commit frequently** - Don't wait until the end of the day
2. **Meaningful commit messages** - Describe what and why, not just what
3. **Logical grouping** - Group related changes together
4. **Update `updates.md` first** - Then commit both together

### Commit Message Format

```
Type: Brief description

- Detailed change 1
- Detailed change 2
- Related files modified
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

**Examples:**
```bash
git commit -m "feat: Add Hugging Face dataset integration

- Created dataset download script
- Added dataset loader with split support
- Implemented training script with proper splits
- Added comprehensive documentation"
```

```bash
git commit -m "docs: Update methodology documentation

- Added dataset statistics
- Documented training hyperparameters
- Added reproducibility guidelines"
```

### Standard Workflow

```bash
# 1. Check status
git status

# 2. Stage changes
git add <specific-files>  # Prefer specific files over git add .

# 3. Update updates.md if needed
# (Edit updates.md)

# 4. Commit with meaningful message
git commit -m "Type: Description"

# 5. Push to remote
git push
```

### Before Committing Checklist

- [ ] Code works (no syntax errors, tested if applicable)
- [ ] `updates.md` is updated with today's work
- [ ] Commit message is clear and descriptive
- [ ] Only relevant files are staged (no temp files, __pycache__, etc.)
- [ ] Large files are in `.gitignore` (checkpoints, datasets, images)

---

## 📁 File Organization

### Directory Structure

```
├── data/                    # Datasets (gitignored)
├── docs/                    # Documentation
├── Notebooks/               # Jupyter notebooks and training scripts
├── runs/                    # Training outputs (gitignored)
├── testing_files/          # Test scripts and results
├── utils/                  # Utility scripts
├── train_clip*.py          # Training scripts
├── inference*.py           # Inference scripts
├── updates.md              # Daily progress log
├── README.md               # Project overview
└── .gitignore              # Git ignore rules
```

### File Naming Conventions

- **Scripts:** `snake_case.py` (e.g., `train_clip_hf_dataset.py`)
- **Documentation:** `UPPER_SNAKE_CASE.md` (e.g., `HF_DATASET_METHODOLOGY.md`)
- **Config files:** `snake_case.json` or `.yaml`
- **Notebooks:** `Descriptive_Name.ipynb`

### What NOT to Create

❌ **Avoid creating:**
- Duplicate files with similar functionality
- Temporary test files (use existing test directories)
- Unnecessary configuration files
- Files that duplicate existing functionality
- One-off scripts that should be integrated into existing files

✅ **Before creating a new file, ask:**
- Does this functionality already exist?
- Can this be added to an existing file?
- Is this a one-time use or reusable?
- Will this be needed long-term?

---

## 🚀 Incremental Progress

### Work in Small Steps

1. **Plan first** - Understand what needs to be done
2. **Implement incrementally** - One feature at a time
3. **Test as you go** - Verify each step works
4. **Commit frequently** - Don't wait for "perfect" code
5. **Document changes** - Update relevant docs

### Example Workflow

```
1. Plan: "I need to add dataset download functionality"
2. Create: `utils/download_hf_chess_dataset.py`
3. Test: Run script, verify it works
4. Document: Add to `updates.md` and relevant docs
5. Commit: "feat: Add HF dataset download script"
6. Move to next feature
```

### When to Refactor

- ✅ When code duplication appears
- ✅ When files become too large (>500 lines)
- ✅ When functionality is reused 3+ times
- ❌ Don't refactor "just because"
- ❌ Don't refactor working code without reason

---

## 📚 Documentation Standards

### When to Update Documentation

- ✅ Adding new features or scripts
- ✅ Changing methodology or approach
- ✅ Fixing bugs that affect functionality
- ✅ Adding new datasets or models
- ✅ Changing training procedures

### Documentation Files

- **`README.md`**: Project overview, setup, quick start
- **`docs/*.md`**: Detailed documentation (methodology, usage, etc.)
- **`updates.md`**: Daily progress log
- **Code comments**: Explain why, not what (code should be self-explanatory)

### Documentation Checklist

- [ ] README updated if user-facing changes
- [ ] Relevant docs updated (methodology, usage, etc.)
- [ ] Code has appropriate comments
- [ ] `updates.md` has entry for the work

---

## 🔍 Code Quality

### Before Committing Code

1. **Check for errors:**
   ```bash
   python -m py_compile <file.py>  # Syntax check
   ```

2. **Review the code:**
   - Is it readable?
   - Are variable names clear?
   - Is it following existing patterns?

3. **Test if applicable:**
   - Does it run without errors?
   - Does it produce expected output?

### Code Style

- Follow existing code patterns in the project
- Use meaningful variable names
- Add docstrings for functions/classes
- Keep functions focused (single responsibility)

---

## 🎯 Session Start Checklist

When starting a new Cursor session:

1. **Read `updates.md`** - Understand recent work
2. **Check `git status`** - See what's changed
3. **Review open issues/tasks** - Understand current priorities
4. **Plan the work** - What needs to be done today?
5. **Start working** - Follow this guide

---

## 🎯 Session End Checklist

Before ending a session:

1. **Update `updates.md`** - Log all work done
2. **Review changes** - `git status` and `git diff`
3. **Commit changes** - With meaningful messages
4. **Push to remote** - `git push`
5. **Verify** - Check that everything is saved

---

## ⚠️ Common Mistakes to Avoid

1. **❌ Forgetting to update `updates.md`**
   - ✅ Always update at end of session

2. **❌ Committing without testing**
   - ✅ Test code before committing

3. **❌ Vague commit messages**
   - ✅ Use descriptive, meaningful messages

4. **❌ Creating duplicate files**
   - ✅ Check if functionality exists first

5. **❌ Committing large files**
   - ✅ Check `.gitignore`, use git-lfs if needed

6. **❌ Working on multiple features at once**
   - ✅ Complete one feature, commit, then move on

7. **❌ Not documenting changes**
   - ✅ Update relevant docs when making changes

---

## 📖 Quick Reference

### Essential Commands

```bash
# Check status
git status

# Stage specific file
git add path/to/file.py

# Commit with message
git commit -m "Type: Description"

# Push to remote
git push

# View recent commits
git log --oneline -10

# Check what changed
git diff
```

### File Locations

- **Progress log:** `updates.md`
- **Project docs:** `docs/`
- **Training scripts:** `train_clip*.py`, `Notebooks/`
- **Utilities:** `utils/`
- **Documentation:** `docs/`, `README.md`

---

## 💡 Tips for AI Assistants

When working in Cursor:

1. **Read context first** - Check `updates.md` and recent files
2. **Ask before creating** - Verify if file/feature already exists
3. **Update docs** - Always update relevant documentation
4. **Commit properly** - Use meaningful commit messages
5. **Test incrementally** - Don't wait until the end
6. **Follow patterns** - Match existing code style and structure
7. **Keep it simple** - Don't over-engineer solutions

---

## 📞 Need Help?

- Check `updates.md` for recent work
- Review `README.md` for project overview
- Check `docs/` for detailed documentation
- Review git history: `git log --oneline`

---

**Remember:** The goal is incremental, well-documented progress. Quality over quantity. Small, tested, committed changes are better than large, untested, uncommitted changes.


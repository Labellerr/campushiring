# Pull Request Submission Guide

## Steps to Submit Your Image Segmentation Project

### 1. Repository Setup

First, ensure you have the forked repository cloned locally:

```bash
# If you haven't cloned your fork yet
git clone https://github.com/YOUR_USERNAME/campushiring.git
cd campushiring

# Add the original repository as upstream
git remote add upstream https://github.com/A-k-s-h-a-t-G-u-p-t-a/campushiring.git
git remote -v  # Verify remotes are set correctly
```

### 2. Create Your Project Directory

**Important**: Replace `your_firstname_lastname` with your actual name (e.g., `john_doe`, `priya_sharma`)

```bash
# Navigate to the repository root
cd campushiring

# Create your directory (replace with your actual name)
mkdir your_firstname_lastname
cd your_firstname_lastname

# Copy your project files here
# - README.md
# - sources.md  
# - requirements.txt
# - Any additional project files/folders
```

### 3. Project Structure Example

Your directory should look like this:
```
campushiring/
└── your_firstname_lastname/           # Replace with your actual name
    ├── README.md                      # Project documentation
    ├── sources.md                     # Data source attributions
    ├── requirements.txt               # Python dependencies
    ├── data/                          # Dataset files (if small)
    ├── models/                        # Model configurations/weights
    ├── src/                           # Source code
    ├── notebooks/                     # Jupyter notebooks
    ├── webapp/                        # Web application files
    └── docs/                          # Additional documentation
```

### 4. Git Commands for Submission

```bash
# Make sure you're in the repository root
cd campushiring

# Create and switch to a new branch for your submission
git checkout -b feature/your-firstname-lastname-submission

# Add all your files
git add your_firstname_lastname/

# Commit your changes with a descriptive message
git commit -m "Add image segmentation project - [Your Name]

- End-to-end YOLO-Seg implementation with Labellerr integration
- ByteTrack object tracking for video sequences
- Complete documentation and source attribution
- Web application for interactive demo"

# Push your branch to your fork
git push origin feature/your-firstname-lastname-submission
```

### 5. Create Pull Request

1. **Go to your forked repository** on GitHub: `https://github.com/YOUR_USERNAME/campushiring`

2. **Click "Compare & pull request"** button (should appear after pushing your branch)

3. **Fill out the PR template**:
   - **Title**: `[Your Name] - Image Segmentation & Object Tracking Project`
   - **Description**: 
     ```
     ## Assignment Submission: Image Segmentation & Object Tracking
     
     **Student**: [Your Full Name]
     **Date**: September 23, 2025
     
     ### Project Overview
     Complete end-to-end implementation of image segmentation and object tracking system using:
     - YOLOv8-Seg for object detection and segmentation
     - Labellerr for data annotation and management
     - ByteTrack for multi-object tracking in videos
     - Web application for interactive demonstration
     
     ### Deliverables Included
     - [x] Complete project documentation (README.md)
     - [x] Data source attribution (sources.md)
     - [x] Python dependencies (requirements.txt)
     - [x] Source code and implementation files
     - [x] Model training and evaluation notebooks
     - [x] Web application for video tracking demo
     
     ### Live Demo
     [Add your deployed demo link here when ready]
     
     ### Key Features
     - Trained on 100+ annotated images (vehicles & pedestrians)
     - Tested on 50+ independent test images
     - Real-time video object tracking
     - JSON export functionality for tracking results
     - Labellerr SDK integration for annotation workflow
     
     Please review and merge this submission for the Computer Vision Development Challenge.
     ```

4. **Set the base repository**: Ensure you're creating the PR to `A-k-s-h-a-t-G-u-p-t-a/campushiring` (not your own fork)

5. **Add reviewers** if specified in the assignment

6. **Click "Create pull request"**

### 6. Post-Submission Checklist

After creating the PR:

- [ ] Verify all files are included in the PR
- [ ] Check that the directory name follows the `firstname_lastname` format
- [ ] Ensure README.md renders correctly on GitHub
- [ ] Verify all links in documentation work
- [ ] Test that requirements.txt includes all necessary dependencies
- [ ] Update the live demo link in the PR description once deployed

### 7. Common Issues & Solutions

**Issue**: "This branch has conflicts that must be resolved"
```bash
# Update your branch with latest changes from upstream
git fetch upstream
git checkout feature/your-firstname-lastname-submission
git merge upstream/main
# Resolve conflicts if any, then push
git push origin feature/your-firstname-lastname-submission
```

**Issue**: "No changes to commit"
```bash
# Make sure you're in the right directory and files are staged
git status
git add .
git commit -m "Your commit message"
```

**Issue**: Large file sizes
- Use Git LFS for large model files
- Consider hosting large datasets externally and providing download links
- Include only essential files in the repository

### 8. Best Practices

1. **Keep commits atomic**: Each commit should represent a logical change
2. **Write clear commit messages**: Explain what and why, not just what
3. **Test locally**: Ensure your code runs before submitting
4. **Document everything**: Clear README, code comments, and attributions
5. **Follow naming conventions**: Use the exact format specified in guidelines

### Need Help?

If you encounter issues:
1. Check the original assignment requirements
2. Review GitHub's pull request documentation
3. Ask for help in the course forum or contact the instructor

---

**Remember**: Replace all placeholder text (like `your_firstname_lastname` and `YOUR_USERNAME`) with your actual information before following these instructions.
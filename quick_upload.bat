@echo off
echo ========================================
echo     AI新生项目GitHub上传脚本
echo ========================================
echo.

echo 请确保您已经：
echo 1. 安装了Git for Windows
echo 2. 在GitHub创建了名为 AI-New-Journey 的仓库
echo 3. 配置了Git用户信息
echo.

pause

echo 开始初始化Git仓库...
git init

echo 添加所有文件...
git add .

echo 创建初始提交...
git commit -m "Initial commit: AI新生教程完整项目 - 包含AI概念、历史时间线、K-Means客户分群等完整实现"

echo 设置默认分支...
git branch -M main

echo.
echo 请输入您的GitHub用户名：
set /p username=

echo 添加远程仓库...
git remote add origin https://github.com/%username%/AI-New-Journey.git

echo 推送到GitHub...
git push -u origin main

echo.
echo ========================================
echo 推送完成！请访问以下链接查看您的项目：
echo https://github.com/%username%/AI-New-Journey
echo ========================================

pause

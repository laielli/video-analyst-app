cd /Users/michaellaielli/career/video-analyst-app

for w in a6ea881786b644fc4 a3fd8d7863253dafa aa75bdf1a3782d4a3 a217a12daf6159107; do
  git branch -D worktree-agent-$w
done
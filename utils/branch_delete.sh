cd /Users/michaellaielli/career/video-analyst-app

for w in abc0dbf79d2233136 ad347149e4ffc4c53 ad44f7e9b70c2d581 afbb61fdfa9bf727d; do
  git branch -D worktree-agent-$w
done
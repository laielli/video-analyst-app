cd /Users/michaellaielli/career/video-analyst-app

for w in a803f5d033bf7ca56 abc0dbf79d2233136 ad44f7e9b70c2d581 ad347149e4ffc4c53 afbb61fdfa9bf727d; do
  git worktree remove --force .claude/worktrees/agent-$w
done
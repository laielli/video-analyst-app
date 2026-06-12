cd /Users/michaellaielli/career/video-analyst-app

for w in a4792aca4b58bc612 a44153b2b64dcc3f6 a98778a8becb96aa1 abbacb39cee8cb0d8 aa3fbac3f994f23ab; do
  git worktree remove --force .claude/worktrees/agent-$w
done
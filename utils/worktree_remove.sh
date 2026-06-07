for w in a1a0e4560669d2e74 a32056960f6e1db28 a53e00fc219b3bb80 a91179c641289c297 ab73ffed980dfc039; do
  git worktree remove --force .claude/worktrees/agent-$w
done
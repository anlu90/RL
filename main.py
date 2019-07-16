import backwards

m = backwards.MAZE
print(m)

a = backwards.create_new_maze(m)
print(a)

backwards.policy(m)

import bellman
m2 = bellman.MAZE2
m1 = bellman.MAZE1
print(m2)
print(m1)

bellman.policy(m1)

bellman.policy(m2)
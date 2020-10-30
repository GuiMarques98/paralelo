import os

bits = 28
os.system(f"make run ppn=1 node=1 > doc/1node1ppn{bits}bits.txt")
os.system(f"make run ppn=1 node=2 > doc/2node1ppn{bits}bits.txt")
os.system(f"make run ppn=1 node=3 > doc/3node1ppn{bits}bits.txt")
os.system(f"make run ppn=2 node=3 > doc/3node2ppn{bits}bits.txt")
os.system(f"make run ppn=4 node=3 > doc/3node4ppn{bits}bits.txt")

os.system(f"make run ppn=8 node=3 > doc/3node8ppn{bits}bits.txt")
os.system(f"make run ppn=12 node=3 > doc/3node12ppn{bits}bits.txt")





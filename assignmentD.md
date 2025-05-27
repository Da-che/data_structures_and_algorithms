# Assignment #D: 图 & 散列表

Updated 2042 GMT+8 May 20, 2025

2025 spring, Complied by <mark>陈宣之 生命科学学院</mark>



> **说明：**
>
> 1. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 2. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 3. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### M17975: 用二次探查法建立散列表

http://cs101.openjudge.cn/practice/17975/

<mark>需要用这样接收数据。因为输入数据可能分行了，不是题面描述的形式。OJ上面有的题目是给C++设计的，细节考虑不周全。</mark>

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
```



思路：



代码：

```python
import sys
input=sys.stdin.read
data=input().split()
ptr=0
n=int(data[ptr])
ptr+=1
m=int(data[ptr])
ptr+=1
num_list=list(map(int,data[ptr:ptr+n]))
ht=[0]*m
ans=[]
for cur in num_list:
    loc=cur%m
    k=1
    if ht[loc]==cur:
        ans.append(loc)
        continue
    while ht[loc]:
        switch=((loc+k*k)%m)
        if not ht[switch]:
            loc=switch
            break
        switch=((loc-k*k)%m)
        if not ht[switch]:
            loc=switch
            break
        k+=1
    ht[loc]=cur
    ans.append(loc)
print(*ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250522114926728](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250522114926728.png)



### M01258: Agri-Net

MST, http://cs101.openjudge.cn/practice/01258/

思路：



代码：

```python
import sys,heapq
data=sys.stdin.read().split()
ptr=0
while ptr<len(data):
    n=int(data[ptr])
    ptr+=1
    matrix=[[int(data[ptr+i*n+j]) for j in range(n)] for i in range(n)]
    ptr+=n*n
    res=0
    dist=[float('inf') for i in range(n)]
    dist[0]=0
    visited=set()
    visited.add(0)
    for _ in range(n-1):
        heap=[]
        for cur in visited:
            for nxt in range(n):
                if nxt not in visited:
                    heapq.heappush(heap,(matrix[cur][nxt],cur,nxt))
        dis,cur,nxt=heapq.heappop(heap)
        res+=dis
        dist[nxt]=dis
        visited.add(nxt)
    print(res)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250522115059911](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250522115059911.png)



### M3552.网络传送门旅游

bfs, https://leetcode.cn/problems/grid-teleportation-traversal/

思路：

gates存传送门，键为字母，值为列表存坐标。bfs，遇到字母将所有传送门坐标加入

代码：

```python
class Solution:
    def minMoves(self, matrix: List[str]) -> int:
        m=len(matrix)
        n=len(matrix[0])
        gates=defaultdict(list)
        for i in range(m):
            for j in range(n):
                if matrix[i][j]!='.' and matrix[i][j]!='#':
                    gates[matrix[i][j]].append((i,j))

        q=deque()
        inq=set()
        q.append((0,0,0))
        inq.add((0,0))
        if matrix[0][0]!='.':
            for gate in gates[matrix[0][0]]:
                if (0,0)==gate:
                    continue
                if gate not in inq:
                    q.append((0,gate[0],gate[1]))
                    inq.add(gate)
        
        while q:
            step,x,y=q.popleft()
            if (x,y)==(m-1,n-1):
                return step
        
            for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx,ny=x+dx,y+dy
                if 0<=nx<m and 0<=ny<n and matrix[nx][ny]!='#' and (nx,ny) not in inq:
                    q.append((step+1,nx,ny))
                    inq.add((nx,ny))
                    if matrix[nx][ny]!='.':
                        for gate in gates[matrix[nx][ny]]:
                            if (nx,ny)==gate:
                                continue
                            if gate not in inq:
                                q.append((step+1,gate[0],gate[1]))
                                inq.add(gate)
        return -1
                    
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250522115423634](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250522115423634.png)



### M787.K站中转内最便宜的航班

Bellman Ford, https://leetcode.cn/problems/cheapest-flights-within-k-stops/

思路：



代码：

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        dist=[float('inf')]*n
        dist[src]=0
        for _ in range(k+1):
            temp=dist.copy()
            for from_loc,to_loc,cost in flights:
                temp[to_loc]=min(temp[to_loc],dist[from_loc]+cost)
            dist=temp.copy()
        if dist[dst]==float('inf'):
            return -1
        else:
            return dist[dst]
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250522123645367](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250522123645367.png)



### M03424: Candies

Dijkstra, http://cs101.openjudge.cn/practice/03424/

思路：

a→b，权重为c的边表示差分约束b-a<=c，边连接为路径代表不等式的相加。用Dijkstra找到从1到n最短的边cost，表示n-1<=cost，即是1和n两者最严格的差分约束，因此在该约束下n-1的最大值即为cost

代码：

```python
import sys,heapq
data=sys.stdin.read().split()
ptr=0
n,m=int(data[ptr]),int(data[ptr+1])
ptr+=2
adj=[[] for i in range(n)]
for i in range(m):
    a,b,weight=int(data[ptr])-1,int(data[ptr+1])-1,int(data[ptr+2])
    adj[a].append((b,weight))  #b-a<=weight to solve (n)-(1)<=x
    ptr+=3
q=[(0,0)]
inq=set()
while q:
    cost,child=heapq.heappop(q)
    inq.add(child)
    if child==n-1:
        print(cost)
        break
    for nxt,weight in adj[child]:
        if nxt not in inq:
            heapq.heappush(q,(cost+weight,nxt))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250522192853256](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250522192853256.png)



### M22508:最小奖金方案

topological order, http://cs101.openjudge.cn/practice/22508/

思路：

拓扑排序，但是通过出度排序，初始化bonus为100，将出度为0的加入队列，出队的u的bonus取其子节点bonus最大值+1，更新父节点出度，若父节点出度为0则入队

代码：

```python
from collections import deque
import sys
data=sys.stdin.read().split()
ptr=0
n=int(data[ptr])
ptr+=1
m=int(data[ptr])
ptr+=1
adj=[[] for _ in range(n)]
rev_adj=[[] for _ in range(n)]
od=[0]*n
bonus=[100]*n
for i in range(m):
    u,v=int(data[ptr]),int(data[ptr+1])
    ptr+=2
    adj[u].append(v)
    rev_adj[v].append(u)
    od[u]+=1

q=deque()
for i in range(n):
    if od[i]==0:
        q.append(i)

while q:
    u=q.popleft()
    bonus[u]=max(bonus[u],max(bonus[v]+1 for v in adj[u]) if adj[u] else 0)
    for v in rev_adj[u]:
        od[v]-=1
        if od[v]==0:
            q.append(v)

print(sum(bonus))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250523161705030](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250523161705030.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这周的作业都是每日选做的题，本来想集中找个时间写，结果发现都做过了hhh。第一题本来是简单的实现，但当时死活想不到到重复出现的数，写的挺头大；第二、四题是最小生成树和Bellman-Ford模板题；Candies第一次见还挺巧妙的，把差分约束变成图变成最短路径题，标签Dijkstra还以为标错了；最后一题是逆向的拓扑排序，正着排似乎不太可行。

图的每日选做上周写完了，准备复习一下数和链表（怎么就要期末了qwq）


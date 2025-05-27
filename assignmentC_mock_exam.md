# Assignment #C: 202505114 Mock Exam

Updated 1518 GMT+8 May 14, 2025

2025 spring, Complied by <mark>陈宣之 生命科学学院</mark>



> **说明：**
>
> 1. **⽉考**：AC<mark>5</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>
> 2. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 3. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### E06364: 牛的选举

http://cs101.openjudge.cn/practice/06364/

思路：

两次sort

代码：

```python
n,k=map(int,input().split())
cows=[]
for _ in range(n):
    cows.append([_]+list(map(int,input().split())))
cows.sort(key=lambda x:x[1],reverse=True)
nxt=cows[:k]
nxt.sort(key=lambda x:x[2],reverse=True)
print(nxt[0][0]+1)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515183441610](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250515183441610.png)



### M04077: 出栈序列统计

http://cs101.openjudge.cn/practice/04077/

思路：

dfs，每一步选择append(curr<=n)或pop(stack非空)，全部出栈(len(out)==n)结束

代码：

```python
n=int(input())
ans=0
def dfs(curr,stack=[],out=[]):
    global ans
    if len(out)==n:
        ans+=1
        return
    if curr<=n:
        dfs(curr+1,stack+[curr],out)
    if stack:
        top=stack.pop()
        dfs(curr,stack,out+[top])
dfs(1)
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515183657702](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250515183657702.png)



### M05343:用队列对扑克牌排序

http://cs101.openjudge.cn/practice/05343/

思路：

入队后按顺序出队

代码：

```python
from collections import deque
n=int(input())
queues=[deque() for _ in range(9)]
cards=list(map(str,input().split()))
for card in cards:
    y=int(card[1])
    queues[y-1].append(card)
for i in range(9):
    print('Queue',i+1,':',' '.join(queues[i]),sep='')

flowers=[deque() for _ in range(4)]
for q in queues:
    while q:
        cur=q.popleft()
        flower=ord(cur[0])-ord('A')
        flowers[flower].append(cur)
for i in range(4):
    print('Queue',chr(i+ord('A')),':',' '.join(flowers[i]),sep='')
ans=[]
for q in flowers:
    while q:
         cur=q.popleft()
         ans.append(cur)
print(*ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515183754427](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250515183754427.png)



### M04084: 拓扑排序

http://cs101.openjudge.cn/practice/04084/

思路：

模板题

代码：

```python
v,a=map(int,input().split())
adj=[[] for _ in range(v)]
ind=[0]*v
for _ in range(a):
    u,w=map(int,input().split())
    adj[u-1].append(w-1)
    ind[w-1]+=1

ans=[]
sorted_ind=[[ind[index],index] for index in range(v)]
sorted_ind.sort()
while sorted_ind[0][0]!=10**9:
    indegee,cur=sorted_ind[0]
    ans.append(cur)
    ind[cur]=10**9
    for nxt in adj[cur]:
        ind[nxt]-=1
    sorted_ind=[[ind[index],index] for index in range(v)]
    sorted_ind.sort()
for i in ans:
    print('v',i+1,sep='',end=' ')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515184009750](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250515184009750.png)



### M07735:道路

Dijkstra, http://cs101.openjudge.cn/practice/07735/

思路：

1.特殊的Dijkstra，除了要找到最短路径，还有路费限制，因此建立二维数组（变成dp了？），记录路费为 j 时到达（i+1）城市的最短距离，初始化后用heap进行bfs（常规Dijkstra），满足条件的节点更新dist并加入q，最后找到dist数组中n城市所有可能路费中的距离最小值，float('inf')表示没有满足条件的路径，print(-1)

2.机考时写的，简单的bfs加heapq，但是机考时用的字典建邻接表，超时或超内存

代码：

```python
1.
import heapq

k = int(input())
n = int(input())
r = int(input())

# 修正：使用列表存储所有边
adj = [[] for _ in range(n)]
for _ in range(r):
    s, d, l, t = map(int, input().split())
    adj[s - 1].append((d - 1, l, t))  # 存储为 (终点, 长度, 费用)

q = [(0, 0, 0)]  # (距离, 费用, 当前城市)
dist = [[float('inf')] * (k + 1) for _ in range(n)]  # dist[i][j]表示到城市i，花费为j的最小距离
dist[0][0] = 0

while q:
    dis, cost, loc = heapq.heappop(q)

    # 跳过过时的状态
    if dis > dist[loc][cost]:
        continue

    # 处理所有可能的路径，而不是找到就立即返回
    for nxt, step_dis, step_cost in adj[loc]:
        new_cost = cost + step_cost
        if new_cost > k:
            continue
        new_dis = dis + step_dis
        if new_dis < dist[nxt][new_cost]:
            dist[nxt][new_cost] = new_dis
            heapq.heappush(q, (new_dis, new_cost, nxt))

# 检查所有可能的费用情况，找到最短路径
ans = float('inf')
for t in range(k + 1):
    if dist[n - 1][t] < ans:
        ans = dist[n - 1][t]

print(ans if ans != float('inf') else -1)
```

```
2.
import heapq
k=int(input())
n=int(input())
r=int(input())
adj=[[] for i in range(n)]
for i in range(r):
    s,d,l,t=map(int,input().split())
    adj[s-1].append((d-1,l,t))

q=[(0,0,0)]
while q:
    dis,cost,loc=heapq.heappop(q)
    if loc==n-1 and cost<=k:
        print(dis)
        break
    for nxt,step_dis,step_cost in adj[loc]:
        if cost+step_cost>k:
            continue
        heapq.heappush(q,(dis+step_dis,cost+step_cost,nxt))
else:
    print(-1)
```

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515194335594](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250515194335594.png)

![image-20250515205434701](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250515205434701.png)

### T24637:宝藏二叉树

dp, http://cs101.openjudge.cn/practice/24637/

思路：

dfs，judge记录父节点的状态，父节点不挖（judge=0），则子节点可挖可不挖（0/1），取最值，父节点挖（1），则节点只能不挖（0），继续搜索两子节点（node*2+1，node*2+2）[完全二叉树]

代码：

```python
n=int(input())
nodes=list(map(int,input().split()))
def dfs(node,judge):
    if node>n-1:
        return 0
    if judge==0:
        return max(dfs(node*2+1,1)+dfs(node*2+2,1)+nodes[node],dfs(node*2+1,0)+dfs(node*2+2,0))
    if judge==1:
        return dfs(node*2+1,0)+dfs(node*2+2,0)
print(dfs(0,0))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515195032559](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250515195032559.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

AC5，本人比较满意，写题颇有成效，但是看到群里都是些一个多小时AK的大佬还是比较震惊。

数组排序，栈，队列排序，图的拓扑排序、最短路径，树的dfs遍历，这次月考几乎把所有数据结构过了一遍，个人感觉整体难度不大，就第五题比较难，但是写的过程当中还是经常会出一些小错误需要debug。

对Dijkstra的理解还停留在上学期bfs的改进，可能还要在仔细看看课件，不然一有变化就无从下手，考试的时候完全套bfs模板，先超时后超内存，问AI发现有多条边起终点相同，因此不能用字典建邻接表。回来后看题解说实话不是太能理解，AI的答案比较好理解。

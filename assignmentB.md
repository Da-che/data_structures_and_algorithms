# Assignment #B: 图为主

Updated 2223 GMT+8 Apr 29, 2025

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

### E07218:献给阿尔吉侬的花束

bfs, http://cs101.openjudge.cn/practice/07218/

思路：

bfs

代码：

```python
from collections import deque

def find_start_end(graph):
    sr,sc,er,ec=0,0,0,0
    for i in range(len(graph)):
        for j in range(len(graph[0])):
            if graph[i][j]=='S':
                sr,sc=i,j
            elif graph[i][j]=='E':
                er,ec=i,j
    return sr,sc,er,ec

def bfs(graph,sr,sc,er,ec):
    q=deque(((sr,sc,0),))
    inq=set()
    inq.add((sr,sc))
    while q:
        x,y,step=q.popleft()
        if x==er and y==ec:
            return step
        for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx,ny=x+dx,y+dy
            if 0<=nx<len(graph) and 0<=ny<len(graph[0]) and graph[nx][ny]!='#' and (nx,ny) not in inq:
                inq.add((nx,ny))
                q.append((nx,ny,step+1))
    return "oop!"

def main():
    t=int(input())
    for i in range(t):
        n,m=map(int,input().split())
        graph=[]
        for _ in range(n):
            graph.append(list(input()))
        sr,sc,er,ec=find_start_end(graph)
        print(bfs(graph,sr,sc,er,ec))

if __name__=="__main__":
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250503171124736](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250503171124736.png)



### M3532.针对图的路径存在性查询I

disjoint set, https://leetcode.cn/problems/path-existence-queries-in-a-graph-i/

思路：

复习一下并查集

代码：

```python
class Solution:
    def pathExistenceQueries(self, n: int, nums: List[int], maxDiff: int, queries: List[List[int]]) -> List[bool]:
        import bisect
        parent=list(range(n))
        def find(x):
            if parent[x]!=x:
                parent[x]=find(parent[x])
            return parent[x]
        def union(x,y):
            px,py=find(x),find(y)
            if px!=py:
                px,py=max(px,py),min(px,py)
                parent[px]=py
        
        def build_disjoint_set(nums,maxDiff):
            n=len(nums)
            for i in range(n-1):
                if nums[i+1]-nums[i]<=maxDiff:
                    union(i,i+1)
        
        def query(u,v):
            if find(u)==find(v):
                return True
            else:
                return False
        
        build_disjoint_set(nums,maxDiff)
        answer=[query(i,j) for i,j in queries]
        return answer
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250504153908015](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250504153908015.png)



### M22528:厚道的调分方法

binary search, http://cs101.openjudge.cn/practice/22528/

思路：

二分查找b值，降序后判断3/5位的数是否>=85

代码：

```python
def score_changer(score,b):
    a=b/1000000000
    return score*a+1.1**(score*a)

def judge(scores):
    scores.sort(reverse=True)
    n=len(scores)
    a=(n*3-1)//5
    if scores[a]>=85:
        return True
    else:
        return False

def main():
    scores=list(map(float,input().split()))
    lft,rgt=0,1000000000
    while lft<=rgt:
        mid=(lft+rgt)//2
        if judge(list(map(lambda x:score_changer(x,mid),scores))):
            rgt=mid-1
        else:
            lft=mid+1
    print(lft)

if __name__=='__main__':
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### Msy382: 有向图判环 

dfs, https://sunnywhy.com/sfbj/10/3/382

思路：

用邻接表建图，然后遍历每个点dfs，visited的跳过，temp存储已经过的节点，若有节点再一次出现则成环

代码：

```python
from collections import defaultdict
class Graph:
    def __init__(self):
        self.graph=defaultdict(list)

    def addEdge(self,u,v):
        self.graph[u].append(v)

def dfs(g,node,temp):
    global visited
    visited[node]=True
    for next in g.graph[node]:
        if next in temp:
            return True
        if dfs(g,next,temp+[next]):
            return True
    return False

g=Graph()
n,m=map(int,input().split())
for _ in range(m):
    u,v=map(int,input().split())
    g.addEdge(u,v)

judge=False
visited=[False]*n
for i in range(n):
    if not visited[i]:
        if dfs(g,i,[i]):
            judge=True
            break
if judge:
    print("Yes")
else:
    print("No")

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250504170313431](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250504170313431.png)



### M05443:兔子与樱花

Dijkstra, http://cs101.openjudge.cn/practice/05443/

思路：

djikstra，除了total_dist之外，还存了path和step_dist记录路径和每一步的距离（开始写dfs习惯了加了inq导致WA，但Dijkstra特点就是找到距离更短的路径后更新该点，允许多次走到该点）

代码：

```python
from collections import defaultdict
import heapq
class Graph:
    def __init__(self):
        self.graph=defaultdict(dict)

    def addEdge(self,location1,location2,distance):
        self.graph[location1][location2]=distance
        self.graph[location2][location1]=distance

    def dijkstra(self,start,end):
        q=[(0,start,[],[])]
        while q:
            total_dist,curr,path,step_dist=heapq.heappop(q)
            if curr==end:
                return path,step_dist
            for neighbor,dist in self.graph[curr].items():
                new_total_dist=total_dist+dist
                new_path=path+[neighbor]
                new_step_dist=step_dist+[dist]
                heapq.heappush(q,(new_total_dist,neighbor,new_path,new_step_dist))

def main():
    g=Graph()
    p=int(input())
    for i in range(p):
        location=input()
        g.graph[location]={}
    q=int(input())
    for i in range(q):
        location1,location2,distance=input().split()
        g.addEdge(location1,location2,int(distance))
    r=int(input())
    for i in range(r):
        start,end=input().split()
        path,step_dist=g.dijkstra(start,end)
        ans=start
        for _ in range(len(path)):
            ans+='->('+str(step_dist[_])+')->'+path[_]
        print(ans)

if __name__=='__main__':
    main()


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250504223127135](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250504223127135.png)



### T28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/

思路：



代码：

```python
最简的dfs写法：超时
def dfs(start,visited,cnt):
    global size
    if cnt==size*size:
        return True
    x,y=start
    for dx,dy in [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]:
        nx,ny=x+dx,y+dy
        if 0<=nx<size and 0<=ny<size and not visited[nx][ny]:
            visited[nx][ny]=True
            if dfs((nx,ny),visited,cnt+1):
                return True
            visited[nx][ny]=False
    return False

size=int(input())
start=tuple(map(int,input().split()))
visited=[[False for _ in range(size)] for _ in range(size)]
if dfs(start,visited,1):
    print("success")
else:
    print("fail")
```



```
从d老师（DeepSeek）学到Warnsdorff规则，优先选择下一步可能性最少的路径
directions = [(1, 2), (2, 1), (-1, 2), (-2, 1),
              (1, -2), (2, -1), (-1, -2), (-2, -1)]

def dfs(start, visited, cnt):
    global size
    if cnt == size * size:
        return True
    x, y = start

    moves = []
    for dx, dy in directions:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < size and 0 <= ny < size and not visited[nx][ny]:
            count = 0
            for ddx, ddy in directions:
                nnx = nx + ddx
                nny = ny + ddy
                if 0 <= nnx < size and 0 <= nny < size and not visited[nnx][nny]:
                    count += 1
            moves.append((count, nx, ny))
    # 按count升序排序
    moves.sort()
    for count, nx, ny in moves:
        visited[nx][ny] = True
        if dfs((nx, ny), visited, cnt + 1):
            return True
        visited[nx][ny] = False
    return False

size = int(input())
sr, sc = map(int, input().split())
visited = [[False] * size for _ in range(size)]
visited[sr][sc] = True
if dfs((sr, sc), visited, 1):
    print("success")
else:
    print("fail")
```

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250505012249562](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250505012249562.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

依旧是搜索为主，dfs、bfs以及Dijkstra，模版背的挺熟了，不同题目细节上有些不一样，debug还是花了些功夫。第二题复习下并查集，上学期一知半解的。最后一题套模板会超时，新学习了一下Warnsdorff规则，好理解，但我确实是想不出来。五一一转眼就过一半了，今天把作业写完，之后几天可以追追每日一练，加油！。








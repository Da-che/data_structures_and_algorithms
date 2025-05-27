# Assignment #A: Graph starts

Updated 1830 GMT+8 Apr 22, 2025

2025 spring, Complied by <mark>同学的姓名、院系</mark>



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

### M19943:图的拉普拉斯矩阵

OOP, implementation, http://cs101.openjudge.cn/practice/19943/

要求创建Graph, Vertex两个类，建图实现。

思路：



代码：

矩阵

```
class Graph:
    def __init__(self,vertices,edges):
        self.vertices=vertices
        self.edges=edges
        self.adj_matrix=self.create_adj_matrix()
        self.degree_matrix=self.create_degree_matrix()

    def create_degree_matrix(self):
        degree_matrix=[[0]*self.vertices for _ in range(self.vertices)]
        for i,j in self.edges:
            degree_matrix[i][i]+=1
            degree_matrix[j][j]+=1
        return degree_matrix

    def create_adj_matrix(self):
        adj_matrix=[[0]*self.vertices for _ in range(self.vertices)]
        for i,j in self.edges:
            adj_matrix[i][j]=1
            adj_matrix[j][i]=1
        return adj_matrix

def main():
    n,m=map(int,input().split())
    edges=[]
    for i in range(m):
        a,b=map(int,input().split())
        edges.append((a,b))
    g=Graph(n,edges)
    laplace_matrix=[[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            laplace_matrix[i][j]=g.degree_matrix[i][j]-g.adj_matrix[i][j]
    for i in range(n):
        print(*laplace_matrix[i])

if __name__=="__main__":
    main()
```

图类

```python
class Vertex:
    def __init__(self,key):
        self.key=key
        self.neighbors={}

    def get_neighbor(self,other):
        return self.neighbors.get(other,None)

    def set_neighbors(self,other,weight=0):
        self.neighbors[other]=weight

    def __str__(self):
        return (str(self.key)
                +'connected to:'
                +str([x.key for x in self.neighbors]))

    def get_neighbors(self):
        return self.neighbors.keys()

    def get_key(self):
        return self.key

class Graph:
    def __init__(self):
        self.vertices={}

    def set_vertex(self,key):
        self.vertices[key]=Vertex(key)

    def get_vertex(self,key):
        return self.vertices.get(key,None)

    def __contains__(self,key):
        return key in self.vertices

    def add_edge(self,from_vertex,to_vertex,weight=0):
        if from_vertex not in self.vertices:
            self.set_vertex(from_vertex)
        if to_vertex not in self.vertices:
            self.set_vertex(to_vertex)
        self.vertices[from_vertex].set_neighbors(self.vertices[to_vertex],weight)

    def get_vertices(self):
        return self.vertices.keys()

    def __iter__(self):
        return iter(self.vertices.values())

def main(n,edges):
    graph=Graph()
    for i in range(n):
        graph.set_vertex(i)

    for edge in edges:
        a,b=edge
        graph.add_edge(a,b)
        graph.add_edge(b,a)

    laplacian=[]
    for vertex in graph:
        row=[0]*n
        row[vertex.get_key()]=len(vertex.get_neighbors())
        for neighbor in vertex.get_neighbors():
            row[neighbor.get_key()]=-1
        laplacian.append(row)

    return laplacian


if __name__=="__main__":
    n,m=map(int,input().split())
    edges=[]
    for i in range(m):
        a,b=map(int,input().split())
        edges.append((a,b))

    laplacian=main(n,edges)
    for row in laplacian:
        print(*row)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429115243094](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250429115243094.png)

![image-20250429115124923](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250429115124923.png)



### LC78.子集

backtracking, https://leetcode.cn/problems/subsets/

思路：

dfs

代码：

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans=[]
        def dfs(prefix=[],index=0):
            if index==len(nums):
                ans.append(prefix)
                return
            dfs(prefix,index+1)
            dfs(prefix+[nums[index]],index+1)
        dfs()
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429115341950](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250429115341950.png)



### LC17.电话号码的字母组合

hash table, backtracking, https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

思路：

dfs

代码：

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        hashtable={'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno',\
                                        '7':'pqrs','8':'tuv','9':'wxyz'}
        ans=[]
        length=len(digits)
        if length==0:
            return ans
        def dfs(step=0,temp=''):
            nonlocal ans,digits
            if step>=length:
                ans.append(temp)
                return
            num=digits[step]
            for dig in hashtable[num]:
                dfs(step+1,temp+dig)
            temp=temp[:-1]
            step-=1
        dfs()
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429131905968](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250429131905968.png)



### M04089:电话号码

trie, http://cs101.openjudge.cn/practice/04089/

思路：

自己还是写不出来，先抄。

代码：

```python
import sys
class TrieNode:
    def __init__(self):
        self.children={}
        self.is_end_of_number=False

class Trie:
    def __init__(self):
        self.root=TrieNode()

    def insert(self,num):
        node=self.root
        for digit in num:
            if digit not in node.children:
                node.children[digit]=TrieNode()
            node=node.children[digit]
            if node.is_end_of_number:
                return False
        node.is_end_of_number=True
        return len(node.children)==0

    def is_prefix(self,numbers):
        numbers.sort()
        for num in numbers:
            if not self.insert(num):
                return False
        return True

def main():
    input=sys.stdin.read()
    data=input.splitlines()

    t=int(data[0])
    index=1
    res=[]

    for i in range(t):
        n=int(data[index])
        index+=1
        numbers=data[index:index+n]
        index+=n
        trie=Trie()
        if trie.is_prefix(numbers):
            res.append("YES")
        else:
            res.append("NO")

    for r in res:
        print(r)

if __name__=="__main__":
    main()

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429184231305](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250429184231305.png)



### T28046:词梯

bfs, http://cs101.openjudge.cn/practice/28046/

思路：

本来用Vertax和Gragh类写，太长了写的脑壳儿昏，遂直接用邻接表。

邻接表建好图后，bfs遍历，predecessor记录路径输出。

代码：

```python
from collections import deque, defaultdict
import sys

n = int(sys.stdin.readline())
words = [sys.stdin.readline().strip() for _ in range(n)]
start, end = sys.stdin.readline().strip().split()

word_set = set(words)
if start not in word_set or end not in word_set:
    print("NO")
    sys.exit()

# 构建邻接表
adj = defaultdict(list)
for word in words:
    neighbors = set()
    for i in range(4):
        current_char = word[i]
        if current_char.isupper():
            letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        else:
            letters = 'abcdefghijklmnopqrstuvwxyz'
        for c in letters:
            if c == current_char:
                continue
            new_word = word[:i] + c + word[i+1:]
            if new_word in word_set:
                neighbors.add(new_word)
    adj[word] = list(neighbors)

# BFS查找最短路径
predecessor = {}
visited = set()
queue = deque([start])
visited.add(start)
predecessor[start] = None
found = False

while queue:
    current = queue.popleft()
    if current == end:
        found = True
        break
    for neighbor in adj[current]:
        if neighbor not in visited:
            visited.add(neighbor)
            predecessor[neighbor] = current
            queue.append(neighbor)

if found:
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessor[current]
    path.reverse()
    print(' '.join(path))
else:
    print("NO")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429201524392](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250429201524392.png)



### T51.N皇后

backtracking, https://leetcode.cn/problems/n-queens/

思路：

用类似全排列的方法，位置代表行，数值代表列，可以避免位于同一行或同一列，主斜线方向行列值差相等，反斜线方向行列值和相等，把差和和存两个集合back_diag和main_diag，可以避免斜向攻击，然后dfs遍历即可。原先将prefix直接存入ans，结果出现浅复制问题（最后全被pop掉了），真的得仔细注意这种细节，错好几回了。

代码：

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        ans=[]
        col_list=[_ for _ in range(n)]
        used=[False]*n
        back_diag=set()
        main_diag=set()

        def dfs(row=0,prefix=[]):
            nonlocal ans
            if len(prefix)==n:
                ans.append(print_format(prefix))
                return
            for col in col_list:
                if not used[col] and row+col not in back_diag and col-row not in main_diag:
                    prefix.append(col)
                    used[col]=True
                    back_diag.add(row+col)
                    main_diag.add(col-row)
                    dfs(row+1,prefix)
                    prefix.pop()
                    used[col]=False
                    back_diag.remove(row+col)
                    main_diag.remove(col-row)
                else:
                    continue
            return

        def print_format(cols):
            final_ans=[]
            for col in cols:
                temp=['.']*n
                temp[col]="Q"
                final_ans.append(''.join(temp))
            return final_ans
        
        dfs()
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429230623246](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250429230623246.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

图写OOP类真的太折磨了，自己暂时还写不顺，希望多写点题加强一下熟练度。dfs和bfs还比较熟，问题不大。Trie类和之前的Huffman树挺有意思的，也是属于看得懂写不出，看看多写几遍能不能记下来，机考还能看看cheat sheet，笔试就不行了。










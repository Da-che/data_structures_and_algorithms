# Cheat Sheet for Data Structure and Algorithms of Chen X.

## 1.排序算法（冒泡、插入、选择略）

### 1.快速排序

```
def quicksort(arr, left, right):
    if left < right:
        partition_pos = partition(arr, left, right)
        quicksort(arr, left, partition_pos - 1)
        quicksort(arr, partition_pos + 1, right)

def partition(arr, left, right):
    i = left
    j = right - 1
    pivot = arr[right]
    while i <= j:
        while i <= right and arr[i] < pivot:
            i += 1
        while j >= left and arr[j] >= pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    if arr[i] > pivot:
        arr[i], arr[right] = arr[right], arr[i]
    return i
```

### 2.归并排序

```
def mergeSort(arr):
	if len(arr) > 1:
		mid = len(arr)//2

		L = arr[:mid]	# Dividing the array elements
		R = arr[mid:] # Into 2 halves

		mergeSort(L) # Sorting the first half
		mergeSort(R) # Sorting the second half

		i = j = k = 0
		# Copy data to temp arrays L[] and R[]
		while i < len(L) and j < len(R):
			if L[i] <= R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				j += 1
			k += 1

		# Checking if any element was left
		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1

		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1
```



## 2.线性结构

### 1.stack中序转后序（Shunting Yard）

```
#Shunting Yard(调度场算法)
def infix_to_postfix(expression):
    precedence = {'+':1, '-':1, '*':2, '/':2}
    stack = []
    postfix = []
    number = ''

    for char in expression:
        if char.isnumeric() or char == '.':
            number += char
        else:
            if number:
                num = float(number)
                postfix.append(int(num) if num.is_integer() else num)
                number = ''
            if char in '+-*/':
                while stack and stack[-1] in '+-*/' and precedence[char] <= precedence[stack[-1]]:
                    postfix.append(stack.pop())
                stack.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()

    if number:
        num = float(number)
        postfix.append(int(num) if num.is_integer() else num)

    while stack:
        postfix.append(stack.pop())

    return ' '.join(str(x) for x in postfix)

n = int(input())
for _ in range(n):
    expression = input()
    print(infix_to_postfix(expression))
```

### 2.stack后序求值

```
def postfixEval(postfixExpr):
    operandStack = []
    tokenList = postfixExpr.split()

    for token in tokenList:
        if token in "0123456789":
            operandStack.append(int(token))
        else:
            operand2 = operandStack.pop()
            operand1 = operandStack.pop()
            result = doMath(token,operand1,operand2)
            operandStack.append(result)
    return operandStack.pop()

def doMath(op, op1, op2):
    if op == "*":
        return op1 * op2
    elif op == "/":
        return op1 / op2
    elif op == "+":
        return op1 + op2
    else:
        return op1 - op2
```



### 3.queue（约瑟夫问题、回文判断）

### 4.差分数组

```
class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n=len(nums)
        diff_array=[0]*(n+1)
        for l,r in queries:
            diff_array[l]+=1
            diff_array[r+1]-=1
        dif=0
        for i in range(n):
            dif+=diff_array[i]
            if nums[i]>dif:
                return False
        return True
        
class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        queries.sort(key=lambda x:x[0])
        deltaArray=[0]*(len(nums)+1)
        heap=[]
        operations=0
        j=0
        for i,num in enumerate(nums):
            operations+=deltaArray[i]
            while j<len(queries) and queries[j][0]==i:
                heappush(heap,-queries[j][1])
                j+=1
            while operations<num and heap and -heap[0]>=i:
                operations+=1
                deltaArray[-heappop(heap)+1]-=1
            if operations<num:
                return -1
        return len(heap)
```



## 3.树

### 1.Traversal

```
def preorder_traversal(root):
    if root:
        print(root.val)  # 访问根节点
        preorder_traversal(root.left)  # 递归遍历左子树
        preorder_traversal(root.right)  # 递归遍历右子树
        
def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)  # 递归遍历左子树
        print(root.val)  # 访问根节点
        inorder_traversal(root.right)  # 递归遍历右子树
        
def postorder_traversal(root):
    if root:
        postorder_traversal(root.left)  # 递归遍历左子树
        postorder_traversal(root.right)  # 递归遍历右子树
        print(root.val)  # 访问根节点
        
def level_order_traversal(root):
    if root is None:
        return []
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        result.append(node.data)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result
```

### 2.建树

```
"""
后序遍历的最后一个元素是树的根节点。然后，在中序遍历序列中，根节点将左右子树分开。
可以通过这种方法找到左右子树的中序遍历序列。然后，使用递归地处理左右子树来构建整个树。
"""
#中后序建树
def buildTree(inorder, postorder):
    if not inorder or not postorder:
        return None

    # 后序遍历的最后一个元素是当前的根节点
    root_val = postorder.pop()
    root = TreeNode(root_val)

    # 在中序遍历中找到根节点的位置
    root_index = inorder.index(root_val)

    # 构建右子树和左子树
    root.right = buildTree(inorder[root_index + 1:], postorder)
    root.left = buildTree(inorder[:root_index], postorder)

    return root
    
#前中序建树
def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_value = preorder[0]
    root = TreeNode(root_value)
    root_index_inorder = inorder.index(root_value)
    root.left = build_tree(preorder[1:1+root_index_inorder], inorder[:root_index_inorder])
    root.right = build_tree(preorder[1+root_index_inorder:], inorder[root_index_inorder+1:])
    return root

#前缀表达式解析树建树
def build_prefix_tree(tokens):
    def helper():
        nonlocal index
        if index >= len(tokens):
            return None
        token = tokens[index]
        index += 1
        node = TreeNode(token)
        if token in '+-*/':
            node.left = helper()
            node.right = helper()
        return node
    
    index = 0
    return helper()

#后缀表达式解析树建树
def build_postfix_tree(tokens):
    stack = []
    for token in tokens:
        if token in '+-*/':
            right = stack.pop()
            left = stack.pop()
            node = TreeNode(token)
            node.left = left
            node.right = right
            stack.append(node)
        else:
            stack.append(TreeNode(token))
    return stack[0]
    
#中缀表达式解析树建树
def build_infix_tree(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '(': 0}
    tokens = expression.replace('(', ' ( ').replace(')', ' ) ').split()
    nodes = []  # 操作数栈
    operators = []  # 运算符栈
    
    for token in tokens:
        if token.isdigit():
            nodes.append(TreeNode(token))
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators[-1] != '(':
                op = operators.pop()
                right = nodes.pop()
                left = nodes.pop()
                node = TreeNode(op)
                node.left = left
                node.right = right
                nodes.append(node)
            operators.pop()  # 弹出 '('
        else:  # 运算符
            while (operators and operators[-1] != '(' and
                   precedence[operators[-1]] >= precedence[token]):
                op = operators.pop()
                right = nodes.pop()
                left = nodes.pop()
                node = TreeNode(op)
                node.left = left
                node.right = right
                nodes.append(node)
            operators.append(token)
    
    # 处理剩余运算符
    while operators:
        op = operators.pop()
        right = nodes.pop()
        left = nodes.pop()
        node = TreeNode(op)
        node.left = left
        node.right = right
        nodes.append(node)
    
    return nodes[0]
```

### 3.Huffman编码树

```
import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(char_freq):
    heap = [Node(char, freq) for char, freq in char_freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq) # note: 合并之后 char 字典是空
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def external_path_length(node, depth=0):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return depth * node.freq
    return (external_path_length(node.left, depth + 1) +
            external_path_length(node.right, depth + 1))
```

### 4.BST

```
def insert(node, value):
    if node is None:
        return TreeNode(value)
    if value < node.value:
        node.left = insert(node.left, value)
    elif value > node.value:
        node.right = insert(node.right, value)
    return node
```

### 5.并查集(Disjoint Set)

```
class UnionFind:
	def __init__(self, n):
		self.Parent = list(range(n))

	def find(self, i):
		if self.Parent[i] != i:
			self.Parent[i] = self.find(self.Parent[i])
		return self.Parent[i]

	def union(self, i, j):
		irep = self.find(i)
		jrep = self.find(j)
		if irep != jrep:
			self.Parent[irep] = jrep
```

### 6.前缀树(Trie)

```
class TrieNode:
    def __init__(self):
        self.children={}
        self.is_end_of_word=False

class Trie:
    def __init__(self):
        self.root=TrieNode()

    def insert(self,word):
        current=self.root
        flag=False
        for char in word:
            if char not in current.children:
                current.children[char]=TrieNode()
            current=current.children[char]
            if current.is_end_of_word:
                flag=True
        current.is_end_of_word=True
        if current.children:
            flag=True
        return flag
```

### 7.树型dp

```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return (0,0)
            l_selected,l_not_selected=dfs(node.left)
            r_selected,r_not_selected=dfs(node.right)
            selected=node.val+l_not_selected+r_not_selected
            not_selected=max(l_selected,l_not_selected)+max(r_selected,r_not_selected)
            return selected,not_selected
        return max(dfs(root))
```



## 4.图

### 1.表示方法（邻接矩阵、邻接表、关联矩阵、Vertex类+Graph类）

### 2.BFS、DFS

### 3.拓扑排序

```
from collections import deque, defaultdict

def topological_sort(graph):
    indegree = defaultdict(int)
    result = []
    queue = deque()
    for u in graph:
        for v in graph[u]:
            indegree[v] += 1
    for u in graph:
        if indegree[u] == 0:
            queue.append(u)
    while queue:
        u = queue.popleft()
        result.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    if len(result) == len(graph):
        return result
    else:
        return None
```

### 4.强连通单元(scc)

```
def dfs1(graph, node, visited, stack):
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs1(graph, neighbor, visited, stack)
    stack.append(node)

def dfs2(graph, node, visited, component):
    visited[node] = True
    component.append(node)
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs2(graph, neighbor, visited, component)

def kosaraju(graph):
    # Step 1: Perform first DFS to get finishing times
    stack = []
    visited = [False] * len(graph)
    for node in range(len(graph)):
        if not visited[node]:
            dfs1(graph, node, visited, stack)
    
    # Step 2: Transpose the graph
    transposed_graph = [[] for _ in range(len(graph))]
    for node in range(len(graph)):
        for neighbor in graph[node]:
            transposed_graph[neighbor].append(node)
    
    # Step 3: Perform second DFS on the transposed graph to find SCCs
    visited = [False] * len(graph)
    sccs = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            scc = []
            dfs2(transposed_graph, node, visited, scc)
            sccs.append(scc)
    return sccs
```

### 5.最短路径

#### 1.Dijkstra（单源无负权）

```
Dijkstra1：
q=[(0,sx,sy)]
inq=set()
inq.add((sx,sy))
while q:
    cost,x,y=heapq.heappop(q)
    inq.add((x,y))
    if (x,y)==(ex,ey):
        print(cost)
        break
    for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        nx,ny=x+dx,y+dy
        if 0<=nx<m and 0<=ny<n and graph[nx][ny]!='#' and (nx,ny) not in inq:
            step_cost=abs(int(graph[nx][ny])-int(graph[x][y]))
            new_cost=cost+step_cost
            heapq.heappush(q,(new_cost,nx,ny))
else:
    print("NO")


Dijkstra2：
min_time={(x,y):float('inf')}
min_time[home]=0
heap=[(0,home[0],home[1])]
while heap:
    time,x,y=heapq.heappop(heap)
    if time>min_time[(x,y)]:
        continue
    if (x,y)==school:
        break
    for nx,ny in min_time.keys():
        if (nx,ny)==(x,y):
            continue
        dist=get_dist(x,y,nx,ny)
        if ((x,y),(nx,ny)) in subway:
            t=dist/40000*60
        else:
            t=dist/10000*60
        if time+t<min_time[(nx,ny)]:
            min_time[(nx,ny)]=time+t
            heapq.heappush(heap,(time+t,nx,ny))
print(round(min_time[school]))
```

#### 2.Bellman-Ford

```
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

```
#检测正权环
dist=[0.0]*currencies
dist[ori_currency]=ori_quantity
for t in range(currencies-1):
    cur_dist=dist.copy()
    for i in range(currencies):
        for to_currency,from_change,from_premium in adj[i]:
            if (dist[i]-from_premium)*from_change>cur_dist[to_currency]:
                cur_dist[to_currency]=(dist[i]-from_premium)*from_change
    dist=cur_dist.copy()

has_positive=False
for i in range(currencies):
    for to_currency,from_change,from_premium in adj[i]:
        if (dist[i]-from_premium)*from_change>dist[to_currency]:
            has_positive=True
            break
    if has_positive:
        break
print("YES" if has_positive else "NO")
```

### 6.最小生成树(MST)

```
def prim(graph, start_node):
    mst = set()
    visited = set([start_node])
    edges = [
        (cost, start_node, to)
        for to, cost in graph[start_node].items()
    ]
    heapify(edges)

    while edges:
        cost, frm, to = heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.add((frm, to, cost))
            for to_next, cost2 in graph[to].items():
                if to_next not in visited:
                    heappush(edges, (cost2, to, to_next))

    return mst
```

### 7.KMP算法

```
""""
compute_lps 函数用于计算模式字符串的LPS表。LPS表是一个数组，
其中的每个元素表示模式字符串中当前位置之前的子串的最长前缀后缀的长度。
该函数使用了两个指针 length 和 i，从模式字符串的第二个字符开始遍历。
"""
def compute_lps(pattern):
    """
    计算pattern字符串的最长前缀后缀（Longest Proper Prefix which is also Suffix）表
    :param pattern: 模式字符串
    :return: lps表
    """

    m = len(pattern)
    lps = [0] * m  # 初始化lps数组
    length = 0  # 当前最长前后缀长度
    for i in range(1, m):  # 注意i从1开始，lps[0]永远是0
        while length > 0 and pattern[i] != pattern[length]:
            length = lps[length - 1]  # 回退到上一个有效前后缀长度
        if pattern[i] == pattern[length]:
            length += 1
        lps[i] = length

    return lps

def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    if m == 0:
        return 0
    lps = compute_lps(pattern)
    matches = []

    # 在 text 中查找 pattern
    j = 0  # 模式串指针
    for i in range(n):  # 主串指针
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]  # 模式串回退
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i - j + 1)  # 匹配成功
            j = lps[j - 1]  # 查找下一个匹配

    return matches
```

**🌟 引理：**

对于某一字符串 $S[1∼i]$，在它的 `next[i]` 的候选值中，若存在某一 `next[i]` 使得：

> 注意这个i是从1开始的，写代码通常从0开始。

$i \mod  (i−next[i])=0$

那么：

- $S[1∼(i−next[i])]$ 是 $S[1∼i]$ 的**最小循环元**（最小周期子串）；
- $K= \frac{i}{i−next[i]}$ 是这个循环元在 $S[1∼i]$ 中出现的次数。
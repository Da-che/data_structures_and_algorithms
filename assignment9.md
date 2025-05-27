# Assignment #9: Huffman, BST & Heap

Updated 1834 GMT+8 Apr 15, 2025

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

### LC222.完全二叉树的节点个数

dfs, https://leetcode.cn/problems/count-complete-tree-nodes/

思路：

dfs遍历

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return 0
            lft,rgt=0,0
            if node.left:
                lft=dfs(node.left)
            if node.right:
                rgt=dfs(node.right)
            return lft+rgt+1
        return dfs(root)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250421184024442](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250421184024442.png)



### LC103.二叉树的锯齿形层序遍历

bfs, https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/

思路：

bfs遍历，每层存入defaultdict

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    from collections import deque,defaultdict
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        q=deque()
        q.append((0,root))
        d=defaultdict(list)
        if not root:
            return []
        while q:
            layer,node=q.popleft()
            d[layer].append(node.val)
            if node.left:
                q.append((layer+1,node.left))
            if node.right:
                q.append((layer+1,node.right))
        
        ans=[]
        layer=0
        while d[layer]:
            if layer%2==0:
                ans.append(d[layer])
            else:
                ans.append(d[layer][::-1])
            layer+=1
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250421190753134](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250421190753134.png)



### M04080:Huffman编码树

greedy, http://cs101.openjudge.cn/practice/04080/

思路：

heapq找到最小的两个数合并

代码：

```python
import heapq
def huffman(arr):
    ans=0
    heapq.heapify(arr)
    while len(arr)>1:
        a,b=heapq.heappop(arr),heapq.heappop(arr)
        ans+=a+b
        heapq.heappush(arr,a+b)
    return ans

if __name__=='__main__':
    n=int(input())
    arr=list(map(int,input().split()))
    print(huffman(arr))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422094252378](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250422094252378.png)



### M05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/

思路：



代码：

```python
from collections import deque
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def bst_build(root,val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = bst_build(root.left,val)
    else:
        root.right = bst_build(root.right,val)
    return root

def level_order_traversal(root):
    if not root:
        return []
    queue=deque()
    queue.append(root)
    res=[]
    while queue:
        node=queue.popleft()
        res.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return res

def main():
    arr=list(map(int,input().split()))
    processed=set()
    root=None
    for i in arr:
        if i in processed:
            continue
        processed.add(i)
        root=bst_build(root,i)
    print(*level_order_traversal(root))

if __name__ == '__main__':
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422101038331](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250422101038331.png)



### M04078: 实现堆结构

手搓实现，http://cs101.openjudge.cn/practice/04078/

类似的题目是 晴问9.7: 向下调整构建大顶堆，https://sunnywhy.com/sfbj/9/7

思路：



代码：

```python
class BinaryHeap:
    def __init__(self):
        self._heap=[]

    def _percolate_up(self, i):
        while (i-1)//2>=0:
            parent_i=(i-1)//2
            if self._heap[i]<self._heap[parent_i]:
                self._heap[i],self._heap[parent_i]=\
                    self._heap[parent_i],self._heap[i]
            i=parent_i

    def _get_min_child(self, i):
        if 2*i+2>len(self._heap)-1:
            return 2*i+1
        if self._heap[2*i+1]<self._heap[2*i+2]:
            return 2*i+1
        return 2*i+2

    def _percolate_down(self, i):
        while 2*i+1<len(self._heap):
            child=self._get_min_child(i)
            if self._heap[i]>self._heap[child]:
                self._heap[i],self._heap[child]=\
                    self._heap[child],self._heap[i]
            else:
                break
            i=child

    def insert(self, val):
        self._heap.append(val)
        self._percolate_up(len(self._heap)-1)

    def delete_min(self):
        self._heap[0], self._heap[-1]=self._heap[-1], self._heap[0]
        res=self._heap.pop()
        self._percolate_down(0)
        return res

def main():
    n=int(input())
    q=BinaryHeap()
    for i in range(n):
        operation=input().split()
        if operation[0]=="1":
            q.insert(int(operation[1]))
        elif operation[0]=="2":
            print(q.delete_min())

if __name__=="__main__":
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422140531492](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250422140531492.png)



### T22161: 哈夫曼编码树

greedy, http://cs101.openjudge.cn/practice/22161/

思路：



代码：

```python
import heapq
import sys

class TreeNode:
    def __init__(self,weight,char=None):
        self.weight=weight
        self.char=char
        self.left=None
        self.right=None

    def __lt__(self, other):
        if self.weight==other.weight:
            return self.char<other.char
        return self.weight<other.weight

def build_tree(char_freq):
    heap=[]
    for char,freq in char_freq.items():
        heapq.heappush(heap,TreeNode(freq,char))
    while len(heap)>1:
        left=heapq.heappop(heap)
        right=heapq.heappop(heap)
        merged=TreeNode(left.weight+right.weight,min(left.char,right.char))
        merged.left=left
        merged.right=right
        heapq.heappush(heap,merged)
    return heap[0]

def encode(root):
    codes={}
    def traverse(node,code):
        if node.left is None and node.right is None:
            codes[node.char]=code
        else:
            traverse(node.left,code+"0")
            traverse(node.right,code+"1")
    traverse(root,'')
    return codes

def decode(codes,string):
    encoded=''
    for char in string:
        encoded+=codes[char]
    return encoded

def huffman_encoding(root,string):
    decoded=''
    node=root
    for bit in string:
        if bit=='0':
            node=node.left
        else:
            node=node.right

        if node.left is None and node.right is None:
            decoded+=node.char
            node=root
    return decoded

def main():
    commands=sys.stdin.readlines()
    n=int(commands[0].strip())
    char_freq={}
    for i in range(1,n+1):
        char,freq=commands[i].strip().split()
        char_freq[char]=int(freq)

    root=build_tree(char_freq)
    codes=encode(root)

    for string in commands[n+1:]:
        string=string.strip()
        if string[0] in ('0','1'):
            print(huffman_encoding(root,string))
        else:
            print(decode(codes,string))

if __name__=='__main__':
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422143631258](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250422143631258.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

BST感觉还好，huffman树和手搓heap还是太复杂了，代码长的吓人，具体原理能理解，真要自己写还是比较困难，照着课件写了几遍，依旧没有自信能顺利写出来。










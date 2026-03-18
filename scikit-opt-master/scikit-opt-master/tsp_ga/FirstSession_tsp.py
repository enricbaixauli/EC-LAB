# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Evolutionary Computation
# **Master in Artificial Intelligence, UVigo, UdC, USC**
# Academic year 2025/26

# Let us import all necessary modules:

import csv
import heapq as hp
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import spatial
from sko.GA import GA_TSP
from sko.PSO import PSO_TSP
import statistics as stats
import sys

# Some of the algorithms allow for a visualization of the intermediate steps. With this variable we can swtich it on or off, whatever might be convenient for you.

showProgress=False
storePng=False

# We initialize a certain number of Monte Carlo rounds.

rounds=2

# We use some boolean variable to turn on or off certain algorithms.

runCI=True  # closest insertion algorithm
runFI=False  # farest insertion algorithm (still not implemented)
runGA=True  # genetic algorithm
runNA=False  # nearest addition algorithm (still not implemented)
runNI=False  # nearest insertion algorithm (still not implemented)
runRI=True  # random insertion algorithm (or quick tour)
runPCs=True # pair-center algorithm (slow version)
runPCf=True # pair-center algorithm (somewhat faster version)
runPCq=True # pair-center algorithm (with ideas from quick tour)
runPSO=False # particle swarm optimization

useLasVegas=False

# The instance file we use as example through out this notebook.

filename='berlin52'

# ### Some helper function
#
# We take as distance the Euclidean distance between two points. However, you can experiment with other metrics, e.g., the Manhatten distance.

def Distance(p,q):
  # Manhatten distance
  #return math.fabs(p[0]-q[0])+math.fabs(p[1]-q[1])
  # Euclidean distance
  return math.hypot(p[0]-q[0],p[1]-q[1])


# For the total tour length we'll take as distance the rounded Euclidean distance between two points which is the standard way for most of the benchmarks.

def DistanceRounded(p,q):
  return int(math.hypot(p[0]-q[0],p[1]-q[1])+0.5)

# A helper function that just computes the side lengths of the triangle built by the three given points.

def TriangleSides(prev,curr,p):
  s0=Distance(prev,curr)
  s1=Distance(prev,p)
  s2=Distance(p,curr)
  return s0,s1,s2


# We read a tour in TSPLIB format. Observe that we assume `csv`-files where the separator is the blank, so maybe you need to modify the original files from TSPLIB and remove all spaces in front of a colon etc.
#
# This function tries to read the corresponding optimal tour as well, whenever available in the same directory.

def ReadTsp(fn):
  f=open(fn+".tsp","rt")
  reader=csv.reader(f,delimiter=" ",skipinitialspace=True)
  coordinates=False
  cities=[]
  best=0.0
  index=0
  for row in reader:
    if row[0]=="EOF":
      break
    elif coordinates:
      cities.append((float(row[1]),float(row[2]),index))
      index+=1
    elif row[0]=="BEST:" or row[0]=="OPTIMUM:":
      best=float(row[1])
    elif row[0]=="BEST" or row[0]=="OPTIMUM":
      best=float(row[2])
    elif row[0]=="NAME:":
      print("reading tour",row[1])
    elif row[0]=="NAME":
      print("reading tour",row[2])
    elif row[0]=="NODE_COORD_SECTION":
      coordinates=True
  f.close()
  opt=[]
  try:
    f=open(fn+".opt.tour","rt")
    reader=csv.reader(f,delimiter=" ",skipinitialspace=True)
    coordinates=False
    for row in reader:
      if row[0]=="EOF" or row[0]=="-1":
        break
      elif coordinates:
        index=int(row[0])
        opt.append((cities[index-1][0],cities[index-1][1],index))
      elif row[0]=="NAME:":
        print("reading optimal tour",row[1])
      elif row[0]=="TOUR_SECTION":
        coordinates=True
  except FileNotFoundError:
    print("optimal tour not available")
  finally:
    f.close()

  print("reading done, optimal tour length",best)
  return cities,best,opt


# Given a filename and a tour, we write a tour in TSPLIB optimal tour format.

def WriteTsp(fn,T):
  f=open(fn+".opt.tour","wt")
  f.write("NAME: "+fn+".opt.tour\n")
  f.write("COMMENT : Length "+str(TourLength(T))+"\n")
  f.write("COMMENT : Found by tsp [Arno Formella]\n")
  f.write("TYPE: TOUR\n")
  f.write("DIMENSION: "+str(len(T))+"\n")
  f.write("TOUR_SECTION\n")
  for t in T:
    f.write(str(t[2]+1)+"\n")
  f.write("-1\n")
  f.write("EOF\n")
  f.close()
  print("writing done")


# We compute the length of a tour by accumulating all distances between neighbors.

def TourLength(T):
  t=len(T)
  if t==0:
    return 0.0
  prev=T[t-1]
  length=0.0
  for i in range(0,t):
    curr=T[i]
    length+=DistanceRounded(prev,curr)
    prev=curr
  return length


# As error we use the relative error of the current length compared to the length of the optimal tour in percent.

def TourError(T,best_length):
  return 100.0*(TourLength(T)-best_length)/best_length


# We want to plot a tour in a certain color and with/without the connecting segments. Whenever a filename is given, we write a corresponding `.png`-image as output.

def PlotTour(T,col,show=False,onlypoints=False,fn=None,store=True):
  if T!=[]:
    fig,ax=plt.subplots(figsize=(9.6,9.6))
    plt.subplots_adjust(bottom=0.0, left=0.0, right=1.0, top=1.0)
    ax.margins(0.01)
    ax.axis('equal')
    for label in (ax.get_xticklabels()+ax.get_yticklabels()):
      label.set_fontsize(16)

    T.append(T[0])
    x,y,i=zip(*T)
    T.pop(-1)
    if onlypoints==True:
      ax.scatter(
        x,y,s=100,color=col,marker="o",linewidth=3
      )
    else:
      ax.plot(
        x,y,color=col,marker="o",linewidth=3,markeredgewidth=3,markersize=10
      )
    if fn!=None and store:
      plt.savefig(fn,transparent=True,bbox_inches='tight')
    if show:
      plt.show()
    else:
      plt.close(fig)


# In order to plot a partial tour (to be able to animate the algorithms) we plot such a tour on top of all cities in a similar way as plotting a complete tour.

def PlotProgressTour(C,T,col,show=False,fn=None,store=True):
  fig,ax=plt.subplots(figsize=(9.6,9.6))
  plt.subplots_adjust(bottom=0.0, left=0.0, right=1.0, top=1.0)
  ax.margins(0.01)
  ax.axis('equal')
  for label in (ax.get_xticklabels()+ax.get_yticklabels()):
    label.set_fontsize(16)

  x,y,i=zip(*C)
  ax.scatter(x,y,s=100,color=col,marker="o",linewidth=3)
  if T!=[]:
    T.append(T[0])
    x,y,i=zip(*T)
    T.pop(-1)
    ax.plot(
      x,y,color="cyan",marker="o",linewidth=1,markeredgewidth=2,markersize=6
    )
  if fn!=None and store:
    plt.savefig(
      fn+"%05d"%(len(T)),transparent=True,bbox_inches='tight'
    )
  if show==True:
    plt.show()
  else:
    plt.close(fig)

# We open the file where we will place statistics.

f=open(filename+"_Errors.txt","wt")

def WriteStats(filename,s,f,errors):
  min_error=min(errors)
  avg_error=stats.mean(errors)
  med_error=stats.median(errors)
  if len(errors)>1:
    stdev_error=stats.stdev(errors)
  else:
    stdev_error=0
  max_error=max(errors)
  print("relative error "+
    "{} tour: min {:.2f}% avg {:.2f}% ({:.2f}) med {:.2f}% max {:.2f}%\\\\\n".format(
      s,min_error,avg_error,stdev_error,med_error,max_error
  ))
  f.write(filename+" & "+s+
    " tour & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\\\n".format(
      min_error,avg_error,stdev_error,med_error,max_error
  ))


# ### The quick tour (or random insertion) algorithm
#
#   idea:
#   - generate a random permutation of the cities
#   - take the first three cities to form a triangle to start with
#   - proceed with the rest of the points in given random order
#     and insert a new city such that the tour increase is minimal
#     among all possible insertions
#   - stop when there is no more city
#   
# First we implement the function that returns the index within the given tour T where an insertion of the point p results in a minimum increase of the overall tour length.
#

def BestInsertionIndex(T,p):
  t=len(T)
  prev=T[t-1]
  curr=T[0]
  s0,s1,s2=TriangleSides(prev,curr,p)
  best_i,shortest=0,s1+s2-s0
  prev=curr
  for i in range(1,t):
    curr=T[i]
    s0,s1,s2=TriangleSides(prev,curr,p)
    increase=s1+s2-s0
    if increase<shortest:
      best_i,shortest=i,increase
    prev=curr
  # print("shortest",shortest,"at",best_i)
  return best_i


# Now we implement the overall quick tour (random insertion) algorithm embedded into a Monte Carlo loop with a certain number of rounds and a Las Vegas condition, so we can stop when we found an optimal tour.
#
# We keep and return the best tour that we found during the Monte Carlo loop.

def QuickTour(Cities,rounds,best_length):
  min_error=sys.float_info.max
  errors=[]
  best_tour=[]
  for j in range(0,rounds): # the Monte Carlo loop
    cities=Cities.copy()       # take a new copy of the cities
    random.shuffle(cities)     # mix them up
    tour=[]
    tour.append(cities.pop(0)) # take the first three as starting triangle
    tour.append(cities.pop(0))
    tour.append(cities.pop(0))

    while cities!=[]:  # while there are still unconnected cities
      PlotProgressTour(
        Cities,tour,"skyblue",show=showProgress,fn="progress_QT",
        store=storePng
      )
      c=cities.pop(0)  # take the next city
      i=BestInsertionIndex(tour,c)  # get tour segment with smallest increase
      tour.insert(i,c) # insert city replacing the segment by two new ones
    PlotProgressTour(
      Cities,tour,"skyblue",show=showProgress,fn="progress_QT",
      store=storePng
    )

    error=TourError(tour,best_length)
    errors.append(error)
    print(j+1,"relative error quick tour: ",error,"%")
    if error<min_error:
      min_error=error
      best_tour=tour.copy()
    if useLasVegas and error==0.0: # the Las Vegas condition
      print("optimum found in",j,"rounds")
      WriteTsp("bestTour",best_tour)
      break
  return best_tour,errors


# Let us run the quick tour algorithm on some example.

if runRI:
  theCities,bestLength,bestTour=ReadTsp(filename)
  if bestLength!=TourLength(bestTour):
    print("strange best tour given",TourLength(bestTour))
  bestT,errors=QuickTour(theCities,rounds,bestLength)
  WriteStats(filename,"quick",f,errors)
  PlotTour(bestT,"skyblue",show=showProgress,fn=filename+"_QuickTour.png")


# ###  The closest-neighbor algorithm
#
# idea:
#   - select a random city
#   - proceed with connecting the closest still unconnected city
#   - stop when there is no more city
#   
# We implement the algorithm embedded into a Monte Carlo loop with a certain number of rounds and a Las Vegas condition so we can stop when we found an optimal tour.
#
# We keep and return the best tour that we found during the Monte Carlo loop.

def ClosestNeighborTour(Cities,rounds,best_length):
  min_error=sys.float_info.max
  errors=[]
  best_tour=[]
  # because we have random tie break, more rounds than there are points
  # can be advantageous
  for j in range(0,rounds): # the Monte Carlo loop
    cities=Cities.copy()
    random.shuffle(cities)
    tour=[]
    tour.append(cities.pop(0))
    while cities!=[]:
      min_dis=sys.float_info.max
      p=tour[0][0],tour[0][1]
      candidates=[]
      for i in range(0,len(cities)):
        q=cities[i][0],cities[i][1]
        dis=Distance(p,q)
        if dis<min_dis:
          min_dis=dis
          candidates=[]
          candidates.append(i)
        elif dis==min_dis:
          candidates.append(i)
      tour.insert(0,cities.pop(candidates[random.randrange(len(candidates))]))
      PlotProgressTour(
        Cities,tour,"red",show=showProgress,fn="progress_CN",
        store=storePng
      )

    error=TourError(tour,best_length)
    errors.append(error)
    print(j+1,"relative error closest neighbor: ",error,"%")
    if error<min_error:
      min_error=error
      best_tour=tour.copy()
    if useLasVegas and error==0.0: # the Las Vegas condition
      print("optimum found in",j,"rounds")
      break
  return best_tour,errors


# Let us run the closest neighbor algorithm on some example.

if runCI:
  theCities,bestLength,bestTour=ReadTsp(filename)
  if bestLength!=TourLength(bestTour):
    print("strange best tour given",TourLength(bestTour))

  bestT,errors=ClosestNeighborTour(theCities,rounds,bestLength)
  WriteStats(filename,"closest-neighbor",f,errors)
  PlotTour(bestT,"red",show=showProgress,fn=filename+"_ClosestNeighbor.png")


# ### The pair-center algorithm
#
# idea:
#   - find closest pair among all pairs of cities
#   - substitute this pair of cities by a center city
#   - proceed until there is only one city left
#   - now we have a binary tree structure
#   - travel through the tree top-down and replace
#     a "center" by its underlying pair taking care of inserting
#     the best possibility
#   - stop when all pairs are handled
#   
# The algorithm uses a data structure of a dictionary to hold the binary tree. Each entry in the dictionary uses as key the index of the current point (either an orignal city or a center being constructed during the progress of the algorithm) and as information a tuple that holds both indices of the points being paired and the coordinates. Hence the tuple `(i,j,x,y)` is the center of points with index `i` and index `j` at location `(x,y)`.

#  We need a function that returns a pair of indices with minimal distance among all possible distances witin the point set L.
#  If there is more than one such pair, a random one of those is returned.

def ClosestPair(L):
  t=len(L)
  best=sys.float_info.max
  pairs=[]
  for i in range(0,t):
    for j in range(i+1,t):
      s0=Distance(L[i],L[j])
      if s0<best:
        best=s0
        pairs=[]
        pairs.append((i,j))
      elif s0==best:
        pairs.append((i,j))
  return pairs[random.randrange(len(pairs))]


# Given a list of indices (keys) into a dictionary we build the corresponding tour.

def MakeTourWithDictionary(I,D):
  T=[]
  for i in I:
    T.append((D[i][2],D[i][3],i))
  return T


# Now we implement the overall pair-center tour algorithm which is a deterministic heuristic algorithm.
# This version is still not very good, it could be improved by handling the closest pairs in a more sophisticated data structure, so the overall runtime can be brought down from cubic to something more efficient. 

def PairCenterTour(Cities,best_length):
  cities=Cities.copy()
  mark=0
  D={}
  for p in Cities:
    D[mark]=(-1,-1,p[0],p[1])
    mark+=1

  # First build the binary tree inserting the corresponding centers
  while len(cities)>1:
    a,b=ClosestPair(cities);
    x,y=0.5*(cities[a][0]+cities[b][0]),0.5*(cities[a][1]+cities[b][1])
    D[mark]=(cities[a][2],cities[b][2],x,y)
    cities[a]=(x,y,mark)
    mark+=1
    cities.pop(b)

  # Run the tree downwards and insert the corresponding pair in
  # the best possible way
  mark-=1
  L=[]
  L.append(D[mark][0])
  PlotProgressTour(
    Cities,MakeTourWithDictionary(L,D),"blue",show=showProgress,
    fn="progress_PC",store=storePng
  )
  L.append(D[mark][1])
  PlotProgressTour(
    Cities,MakeTourWithDictionary(L,D),"blue",show=showProgress,
    fn="progress_PC",store=storePng
  )
  t=len(Cities)
  for m in range(mark-1,t-1,-1):
    i=L.index(m)
    Di0=D[L[i]][0]
    Di1=D[L[i]][1]
    q=(D[Di0][2],D[Di0][3])
    r=(D[Di1][2],D[Di1][3])
    p=s=(0,0)
    if i==len(L)-1:
      p=(D[L[i-1]][2],D[L[i-1]][3])
      s=(D[L[0]][2],D[L[0]][3])
    elif i==0:
      p=(D[L[len(L)-1]][2],D[L[len(L)-1]][3])
      s=(D[L[1]][2],D[L[1]][3])
    else:
      p=(D[L[i-1]][2],D[L[i-1]][3])
      s=(D[L[i+1]][2],D[L[i+1]][3])

    if Distance(p,q)+Distance(r,s)>Distance(p,r)+Distance(q,s):
      L[i]=D[m][1]
      L.insert(i+1,D[m][0])
    else:
      L[i]=D[m][0]
      L.insert(i+1,D[m][1])
    PlotProgressTour(
      Cities,MakeTourWithDictionary(L,D),"blue",show=showProgress,
      fn="progress_PC",store=storePng
    )

  tour=MakeTourWithDictionary(L,D)
  error=TourError(tour,best_length)
  errors=[]
  errors.append(error)
  return tour,errors


# Let's run the pair-center algorithm on some example:

if runPCs:
  theCities,bestLength,bestTour=ReadTsp(filename)
  if bestLength!=TourLength(bestTour):
    print("strange best tour given",TourLength(bestTour))

  bestT,errors=PairCenterTour(theCities,bestLength)
  WriteStats(filename,"pair-center (slow)",f,errors)
  PlotTour(bestT,"blue",show=showProgress,fn=filename+"_PairCenterSlow.png")


# ### A somewhat faster pair-center algorithm
#
# idea:
#   - find closest pair among all pairs of cities with a spatial data structure (kd-tree) and neighbor lists
#   - substitute this pair of cities by a center city
#   - proceed until there is only one city left
#   - now we have a binary tree structure
#   - travel through the tree top-down and replace
#     a "center" by its underlying pair taking care of inserting
#     the best possibility
#   - stop when all pairs are handled
#   
# The algorithm uses a data structure of a dictionary to hold the binary tree. Each entry in the dictionary uses as key the index of the current point (either an orignal city or a center being constructed during the progress of the algorithm) and as information a tuple that holds both indices of the points being paired and the coordinates. Hence the tuple `(i,j,x,y)` is the center of points with index `i` and index `j` at location `(x,y)`.

# We calculate the interpoint distances with a spatial data structure.
def LowerBoundFast(C,pq,N):
  kD=spatial.KDTree([(c[0],c[1]) for c in C])
  lower=0.0
  for c in C:
    nn=kD.query((c[0],c[1]),k=2)
    min_dis=int(nn[0][1]+0.5)
    min_j=nn[1][1]
    hp.heappush(pq,(min_dis,c[2],min_j))
    N[min_j].append(c[2])
    lower+=min_dis
  return lower


# Some more helper function for a faster pair-center algorithm.

# +
def MakeTourWithTree(tree):
  T=[]
  i=0
  for t in tree:
    if t[4]:
      T.append((t[2],t[3],i))
      i+=1
  return T

def MakeTourWithList(L,tree):
  T=[]
  i=0
  for l in L:
    T.append((tree[l][2],tree[l][3],i))
    i+=1
  return T

def HandleNeighbors(tree,N,pq,l,x,y):
  min_dis=sys.float_info.max
  min_j=-1
  for j in range(len(tree)-1):
    if l!=j and tree[j][4]:
      dis=Distance((tree[j][2],tree[j][3]),(x,y))
      if dis<min_dis:
        min_dis=dis
        min_j=j
  if min_j!=-1:
    hp.heappush(pq,(min_dis,l,min_j))
    N[min_j].append(l)


# -

# Now we implement the overall faster pair-center tour algorithm which is also a deterministic heuristic algorithm.
# This version is somewhat faster, it still could be improved easily regarding the tour length.

def PairCenterTourFast(Cities,best_length,pq,N):
  tree=[(-1,-1,p[0],p[1],True) for p in Cities]

  # First build the binary tree inserting the corresponding centers
  n=len(tree)
  r=0
  while r<n-1:
    d,a,b=hp.heappop(pq)
    if tree[a][4] and tree[b][4]:
      x,y=0.5*(tree[a][2]+tree[b][2]),0.5*(tree[a][3]+tree[b][3])
      tree.append((a,b,x,y,True))
      tree[a]=(tree[a][0],tree[a][1],tree[a][2],tree[a][3],False)
      tree[b]=(tree[b][0],tree[b][1],tree[b][2],tree[b][3],False)
      HandleNeighbors(tree,N,pq,n+r,x,y)
      for l in N[a]:
        if tree[l][4]:
          x,y=tree[l][2],tree[l][3]
          HandleNeighbors(tree,N,pq,l,x,y)
      for l in N[b]:
        if tree[l][4]:
          x,y=tree[l][2],tree[l][3]
          HandleNeighbors(tree,N,pq,l,x,y)
      r+=1
  assert len(tree)==2*n-1

  # Run the tree downwards and insert the corresponding pair in
  # the best possible way
  L=[]
  a=tree[-1][0]
  b=tree[-1][1]
  L.append(a)
  L.append(b)
  PlotProgressTour(
    Cities,MakeTourWithTree(tree),"blue",show=showProgress,
    fn="progress_PF",store=storePng
  )
  t=len(tree)-2
  for m in range(t,n-1,-1):
    i0,i1=tree[m][0],tree[m][1]
    q=(tree[i0][2],tree[i0][3])
    r=(tree[i1][2],tree[i1][3])
    i=L.index(m)
    if i==len(L)-1:
      p=(tree[L[i-1]][2],tree[L[i-1]][3])
      s=(tree[L[0]][2],tree[L[0]][3])
    elif i==0:
      p=(tree[L[len(L)-1]][2],tree[L[len(L)-1]][3])
      s=(tree[L[1]][2],tree[L[1]][3])
    else:
      p=(tree[L[i-1]][2],tree[L[i-1]][3])
      s=(tree[L[i+1]][2],tree[L[i+1]][3])

    if Distance(p,q)+Distance(r,s)>Distance(p,r)+Distance(q,s):
      a,b=tree[m][1],tree[m][0]
    else:
      a,b=tree[m][0],tree[m][1]
    L[i]=a
    L.insert(i+1,b)
    PlotProgressTour(
      Cities,MakeTourWithList(L,tree),"blue",show=showProgress,
      fn="progress_PF",store=storePng
    )
  tour=MakeTourWithList(L,tree)
  error=TourError(tour,best_length)
  errors=[]
  errors.append(error)
  return tour,errors


# Run the faster pair-center algorithm.

if runPCf:
  pq=[]
  N=[[] for i in range(2*len(theCities))]
  lowerBound=LowerBoundFast(theCities,pq,N)
  print("lower bound",lowerBound)
  bestT,errors=PairCenterTourFast(theCities,bestLength,pq.copy(),N.copy())
  WriteStats(filename,"pair-center (fast)",f,errors)
  PlotTour(bestT,"blue",show=showProgress,fn=filename+"_PairCenterFast.png")


# Now we implement the overall faster and better pair-center tour algorithm which is still a deterministic heuristic algorithm.
# This version is somewhat better, it still could be improved easily regarding the tour length.
#
# We use the quick tour idea to improve the basic pair-center algorithm.

def PairCenterTourQuick(Cities,best_length,pq,N):
  tree=[(-1,-1,p[0],p[1],True) for p in Cities]
  n=len(tree)
  r=0
  while r<n-1:
    d,a,b=hp.heappop(pq)
    if tree[a][4] and tree[b][4]:
      x,y=0.5*(tree[a][2]+tree[b][2]),0.5*(tree[a][3]+tree[b][3])
      tree.append((a,b,x,y,True))
      tree[a]=(tree[a][0],tree[a][1],tree[a][2],tree[a][3],False)
      tree[b]=(tree[b][0],tree[b][1],tree[b][2],tree[b][3],False)
      HandleNeighbors(tree,N,pq,n+r,x,y)
      for l in N[a]:
        if tree[l][4]:
          x,y=tree[l][2],tree[l][3]
          HandleNeighbors(tree,N,pq,l,x,y)
      for l in N[b]:
        if tree[l][4]:
          x,y=tree[l][2],tree[l][3]
          HandleNeighbors(tree,N,pq,l,x,y)
      r+=1
  assert len(tree)==2*n-1

  # Run the tree downwards and insert the corresponding pair in
  # the best possible way
  L=[]
  tour=[]
  a=tree[-1][0]
  b=tree[-1][1]
  L.append(a)
  L.append(b)
  t=0
  tour.append((tree[a][2],tree[a][3],t)); t+=1
  tour.append((tree[b][2],tree[b][3],t)); t+=1
  PlotProgressTour(
    Cities,tour,"blue",show=showProgress,fn="progress_QP",store=storePng
  )
  t=len(tree)-2
  pro=0
  for m in range(t,n-1,-1):
    pro+=1
    if pro%500==0:
      print(pro,"QP second done")
    i0,i1=tree[m][0],tree[m][1]
    i=L.index(m)

    tour[i]=(tree[i0][2],tree[i0][3],tour[i][2])
    c=(tree[i1][2],tree[i1][3],t); t+=1
    j=BestInsertionIndex(tour,c)
    tour.insert(j,c)
    L[i]=tree[m][0]
    L.insert(j,tree[m][1])
    PlotProgressTour(
      Cities,tour,"blue",show=showProgress,fn="progress_QP",store=storePng
    )
  error=TourError(tour,best_length)
  errors=[]
  errors.append(error)
  return tour,errors


# Run the improved pair-center algorithm.

if runPCq:
  bestT,errors=PairCenterTourQuick(theCities,bestLength,pq,N)
  WriteStats(filename,"pair-center (quick)",f,errors)
  PlotTour(bestT,"blue",
    show=showProgress,fn=filename+"_PairCenterQuick.png"
  )


# For a more sophisticated and much faster implemention of the pair-center algorithm take a look into the publication.

# ### The genetic algorithm tour

# Given a list of indices into the list of location, we build the corresponding tour.

def MakeTourWithCities(I,C):
  return [(C[i][0],C[i][1],i) for i in I]


# We implement the computation of a tour with the genetic algorithm from the `scikit-opt` package.

def GeneticAlgorithmTour(cities,best_length):
  n=len(cities)
  points_coordinate=[(l[0],l[1]) for l in cities]

  dist_mat=spatial.distance.cdist(
    points_coordinate, points_coordinate, metric='euclidean'
  )

  def cal_total_distance(routine):
      '''The objective function. input routine, return total distance.
      cal_total_distance(np.arange(n))
      '''
      n,=routine.shape
      return sum([dist_mat[routine[i%n], routine[(i+1)%n]] for i in range(n)])

  min_error=sys.float_info.max
  errors=[]
  best_tour=[]
  for j in range(rounds):
    ga_tsp=GA_TSP(
      func=cal_total_distance, n_dim=n, size_pop=50, max_iter=1000, prob_mut=1
    )
    best_points, best_distance=ga_tsp.run()
    tour=MakeTourWithCities(best_points,theCities)
    error=TourError(tour,bestLength)
    errors.append(error)
    print(j+1,"relative error GA tour: ",error,"%")
    if error<min_error:
      min_error=error
      best_tour=tour.copy()
    if useLasVegas and error==0.0: # the Las Vegas condition
      print("optimum found in",j,"rounds")
      WriteTsp("bestTour",best_tour)
      break
  return best_tour,errors


if runGA:
  theCities,bestLength,bestTour=ReadTsp(filename)
  if bestLength!=TourLength(bestTour):
    print("strange best tour given",TourLength(bestTour))

  bestT,errors=GeneticAlgorithmTour(theCities,bestLength)
  WriteStats(filename,"genetic algorithm",f,errors)
  PlotTour(bestT,"green",show=showProgress,fn=filename+"_GeneticAlgorithm.png")


def PSOTour(cities,best_length):
  n=len(cities)
  points_coordinate=[(l[0],l[1]) for l in cities]

  dist_mat=spatial.distance.cdist(
    points_coordinate, points_coordinate, metric='euclidean'
  )

  def cal_total_distance(routine):
      '''The objective function. input routine, return total distance.
      cal_total_distance(np.arange(n))
      '''
      n,=routine.shape
      return sum([dist_mat[routine[i%n], routine[(i+1)%n]] for i in range(n)])

  min_error=sys.float_info.max
  errors=[]
  best_tour=[]
  #for j in range(rounds):
  iters=100
  j=0
  while iters<2000:
    pso_tsp=PSO_TSP(
      func=cal_total_distance, n_dim=n, size_pop=100, max_iter=iters,
      w=0.8, c1=0.3, c2=0.3
    )
    iters+=100
    best_points, best_distance=pso_tsp.run()
    tour=MakeTourWithCities(best_points,theCities)
    error=TourError(tour,bestLength)
    errors.append(error)
    print(j+1,"relative error PSO tour: ",error,"%")
    j+=1
    if error<min_error:
      min_error=error
      best_tour=tour.copy()
    if useLasVegas and error==0.0: # the Las Vegas condition
      print("optimum found in",j,"rounds")
      WriteTsp("bestTour",best_tour)
      break
  return best_tour,errors


if runPSO:
  theCities,bestLength,bestTour=ReadTsp(filename)
  if bestLength!=TourLength(bestTour):
    print("strange best tour given",TourLength(bestTour))

  bestT,errors=PSOTour(theCities,bestLength)
  WriteStats(filename,"particle swarm",f,errors)
  PlotTour(bestT,"green",show=showProgress,fn=filename+"_ParticleSwarm.png")


# We calculate the complete matrix of interpoint distances and compute the simple lower bound for the length of a tour. We plot both: the cities and the best tour from the input file.

def LowerBound(M,n):
  if len(M)!=n*n:
    print("usage error in LowerBound")
    exit()
  lower=0.0
  for i in range(0,n):
    min_index=sys.float_info.max
    for j in range(0,n):
      if i!=j:
        curr=M[i*nloc+j]
        if curr<min_index:
          min_index=curr
    lower+=min_index
  return lower


# Some lower bound computation.

# +
theCities,bestLength,bestTour=ReadTsp(filename)
if bestLength!=TourLength(bestTour):
  print("strange best tour given",TourLength(bestTour))

nloc=len(theCities)
Dmat=[
  DistanceRounded((theCities[i][0],theCities[i][1]),(theCities[j][0],theCities[j][1]))
    for i in range(0,nloc) for j in range(0,nloc)
  ]
print("lower bound",LowerBound(Dmat,nloc))
PlotTour(theCities,"magenta",show=showProgress,onlypoints=True,fn=filename+"_Cities.png")

# show best tour
print("best tour length",bestLength)
PlotTour(bestTour,"magenta",show=showProgress,fn=filename+"_BestTour")
# -



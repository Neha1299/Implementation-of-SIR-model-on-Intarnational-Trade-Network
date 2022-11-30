
import pandas as pd
import networkx as nx
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

df = pd.read_csv('./trade.csv')
df1 = df[['ReporterName', 'PartnerName','Export']]
df1=df1.values.tolist()
#print(df1)
print(type(df1))
list1=df1
lel_list = len(list1)
for i in list1:
    if 'Unspecified' in i:
        i[2]=0
    for j in list1[list1.index(i):lel_list]:
        if ([i[0],i[1]]== [j[1],j[0]]):
         #print(i[0],i[1],j[1],j[0])
         if(i[2]>j[2]):
             i[2]=i[2]+j[2]
             j[2]=0
         else:
             j[2]+=i[2]
             i[2]=0 
#print(i,j)
df_len = len(df.index)
#print(df[['ReporterName']][1:10])
G = nx.Graph()
for i in range(11999,12229):
    G.add_edge(df.loc[i]['ReporterName'], df.loc[i]['PartnerName'], weight = df.loc[i]['Export'])
    
#G = nx.from_pandas_edgelist(df1[0:100], source= "ReporterName", target= "PartnerName",edge_attr="TradeValue_in_1000_USD",create_using= nx.DiGraph)
exxl= [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 100000000]
exl= [(u, v) for (u, v, d) in G.edges(data=True) if 100000000 > d["weight"] >= 10000000]
el = [(u, v) for (u, v, d) in G.edges(data=True) if 10000000 > d["weight"] >= 1000000]
em=[(u, v) for (u, v, d) in G.edges(data=True) if 1000000 > d["weight"] >= 100000]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if 100000> d["weight"] >= 10000]
exs = [(u, v) for (u, v, d) in G.edges(data=True) if  d["weight"] < 10000]
figure(figsize=(80,80))
pos = nx.circular_layout(G)  # positions for all nodes - seed for reproducibility
# nodes
nx.draw_networkx_nodes(G,pos, node_size=100,node_color='red')
# edges
nx.draw_networkx_edges(G,pos, edgelist=exxl, width=6 , edge_color="red")
nx.draw_networkx_edges(G,pos, edgelist=exl, width=4 , edge_color="black")
nx.draw_networkx_edges(G,pos, edgelist=el, width=3 , edge_color="blue")
nx.draw_networkx_edges(G,pos, edgelist=em, width=2 , edge_color="orchid")
nx.draw_networkx_edges( G,pos,edgelist=esmall, width=1, alpha=1, edge_color="orange",style='dashed')
nx.draw_networkx_edges( G,pos,edgelist=exs, width=1, alpha=0.5, edge_color="aqua",style='dashed')
# node labels
nx.draw_networkx_labels(G,pos, font_size=40, font_family="sans-serif")
# edge weight labels
#edge_labels = nx.get_edge_attributes(G, "weight")
#nx.draw_networkx_edge_labels(G,pos, edge_labels,font_size=10)

#export=nx.get_edge_attributes(G,'TradeValue_in_1000_USD')
#print(export)
num_nodes = nx.number_of_nodes(G)
num_edges = nx.number_of_edges(G)
density = nx.density(G)
transitivity = nx.transitivity(G)
avg_clustering = nx.average_clustering(G)
print("Number of Nodes: %s" % num_nodes)
print("Number of Edges: %s" % num_edges)
print("Density: %s" % density)
#print("Transitivity: %s" % transitivity)
#print("Avg. Clustering: %s" % avg_clustering)
#pos=nx.circular_layout(G)
#print(nx.degree_centrality(G))
 
centrality = nx.eigenvector_centrality(G, max_iter=100, tol=1e-06, nstart=None, weight='Export')
#centrality = nx.centrality(G,weight='Export')
#y=(['%s %0.2f'%(node,centrality[node]) for node in centrality])
y = dict(sorted(centrality.items(), key = lambda item: item[1],reverse=True))
#print(y)
eigen_centrality = [(k,v) for k,v in y.items()]
#print(eigen_centrality)
#print(type(eigen_centrality))
k6=[]
k5=[]
k4=[]
k3=[]
k2=[]
k1=[]
for i in eigen_centrality:
    if i[1] >= 0.0776:
        k6.append(i[0])
    elif 0.0776 > i[1] >= 0.074:
        k5.append(i[0])
    elif 0.074 > i[1] >= 0.069:
        k4.append(i[0])
    elif 0.069 > i[1] >= 0.064:
        k3.append(i[0])
    elif 0.064 > i[1] >= 0.054:
        k2.append(i[0])
    elif 0.054 > i[1]:
        k1.append(i[0])
print('k6',k6, len(k6))
print('k5',k5, len(k5))
print('k4',k4, len(k4))
print('k3',k3, len(k3))
print('k2',k2, len(k2))
print('k1',k1, len(k1))
print('k6', len(k6))
print('k5', len(k5))
print('k4', len(k4))
print('k3', len(k3))
print('k2', len(k2))
print('k1', len(k1))
plt.show()


Infected_country=input('Enter infected country :  ')


if Infected_country in k6:
    print('k6')
    beta = 0.012
    gamma = 0.1
    delta=0.3
elif Infected_country in k5:
    print('k5')
    beta = 0.009
    gamma = 0.15
    delta=0.2
elif Infected_country in k4:
    print('k4')
    beta = 0.007
    gamma = 0.2
    delta=0.1
elif Infected_country in k3:
    print('k3')
    beta = 0.005
    gamma = 0.3
    delta=0.1
elif Infected_country in k2:
    print('k2')
    beta = 0.004
    gamma = 0.42
    delta=0.1
elif Infected_country in k1:
    print('k1')
    beta = 0.0035
    gamma = 0.5
    delta= 0.1
#N= S + I + R
print(num_nodes)
n = 20000
I0= 1
S0 = num_nodes - I0
R0 = num_nodes - (S0 + I0)
t0=0
tn=20000

#Equations
def sus(t,S,I,R):   
    return -beta*S*I + (delta*R)

def inf(t,S,I):
     return beta*S*I-gamma*I

def rec(t,I,R):
     return gamma*I - (delta*R)

#rk4 method
def rk4(t0,S0,I0,R0,tn,n):


   #step size
   h=0.001
   S_list = []
   I_list = []
   R_list = []

   for i in range(1,n):
      k1= h * (sus(t0, S0, I0, R0))
      l1= h * (inf(t0, S0, I0))
      m1= h * (rec(t0, I0,R0))
      
      k2= h * (sus((t0+h/2), (S0+h*k1/2) ,(I0+h*k1/2),(R0+h*k1/2)))
      l2= h * (inf((t0+h/2), (S0+h*l1/2) ,(I0+h*l1/2)))
      m2= h * (rec((t0+h/2),(I0+h*m1/2),(R0+h*m1/2)))
      
      k3= h * (sus((t0+h/2), (S0+h*k2/2) ,(I0+h*k2/2),(R0+h*k2/2)))
      l3= h * (inf((t0+h/2), (S0+h*l2/2) ,(I0+h*l2/2)))
      m3= h * (rec((t0+h/2),(I0+h*m2/2),(R0+h*m2/2)))
      
      k4 = h * (sus((t0+h),(S0+h*k3), (I0+h*k3),(R0+h*k3)))
      l4 = h * (inf((t0+h),(S0+h*l3), (I0+h*l3)))
      m4 = h * (rec((t0+h),(I0+h*m3),(R0+h*m3)))

      k = (k1+2*k2+2*k3+k4)/6
      Sn= S0 + k
      S0=Sn
   
      l = (l1+2*l2+2*l3+l4)/6
      In=I0 + l
      I0=In
       
      m = (m1+2*m2+2*m3+m4)/6
      Rn= R0 + m
      R0 =Rn
      
      t0=t0+h

      S_list.append(S0)
      I_list.append(I0)
      R_list.append(R0)
        
      if((S0 < 0) or (I0 < 0) or (R0 < 0)):
         S0 = S0 - k
         I0 = I0 - l
         R0 = R0 - m
         break

    
    
   peak_infections_index = I_list.index(max(I_list))
   total = S0 + I0 + R0
   text_to_add = "Peak infections = " + str(round(max(I_list), 2)) + " ; at time = " + str(peak_infections_index)    

   print('\nAt t=%.0f, S=%.0f' %((tn/50),Sn))
   print('\nAt t=%.0f, I=%.0f' %((tn/50),In))
   print('\nAt t=%.0f, R=%.0f' %((tn/50),Rn))
   
   #print('\nAt t=',i,' S=',S0,' I=',I0,' R=',R0)
   #rint(S0+I0+R0)
   ax = plt.axes()
   points = [0, 2000, 4000, 6000, 8000, 10000,12000,14000,16000,18000,20000]
   labels = ["0", "40", "80", "120", "160", "200","240","280","320","360","400"]
   plt.setp(ax, xticks = points, xticklabels = labels)
   plt.plot(list(range(len(S_list))), S_list, color = 'aqua')       
   plt.plot(list(range(len(I_list))), I_list, color = 'orchid')
   plt.plot(list(range(len(R_list))), R_list, color = 'green')
   plt.hlines(y = max(I_list), xmin = 0, xmax = peak_infections_index, color = 'b', linestyle = 'dashed', linewidth = 0.5)
   plt.vlines(x = peak_infections_index, ymin = 0, ymax = max(I_list), color = 'b', linestyle = 'dashed', linewidth = 0.5)
   plt.gca().legend(['Susceptible','Infected','Recovered'])
   plt.title(str(Infected_country))

#rk4 method call
rk4(t0,S0,I0,R0,tn,n)

plt.show()   



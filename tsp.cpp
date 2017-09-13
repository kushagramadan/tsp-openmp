#include <iostream>
#include <stdio.h>
#include <fstream>
#include <cstdlib>
#include <cfloat>
#include <string>
#include <math.h>
#include <map>
#include <vector>
#include <algorithm>
#include <queue>
#include <set>
#include <omp.h>

using namespace std;

#define NUM_ITER 10000
#define CROSS_PROB 0.8
#define PMX_PROB 0.2
#define GX_PROB 0.5
#define CX_PROB 0.1
#define ERX_PROB 0.2
#define MUT_PROB 0.1

void printString(char* s, int dim)
{
    for(int i=0; i<dim; i++)
    {
        cout<<s[i];
    }
    cout<<endl;
}

void strcopy(char* dest, char* src, int dim)
{
    for(int i=0; i<dim; i++)
    {
        dest[i] = src[i];
    }
}

int index(char c)
{
    if(isalpha(c))
    {
        return (int)(c-97);
    }
    else
    {
        return ((int)(c - '0') + 26);
    }
}

double distance(double x1, double y1, double x2, double y2)
{
    return sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
}

double evaluateFitness(char* s, int dim, double** dist)
{
    double ret = 0;
    for(int i=0; i<dim; i++)
    {
        ret += dist[index(s[i%dim])][index(s[(i+1)%dim])];
    }
    return (1/ret);
}

void pmx(char* p1, char* p2, int dim, char* ret)
{
    int rand1 = rand() % dim;
    int rand2 = rand() % dim;
    while(rand1 == rand2)
    {
        rand2 = rand() % dim;
    }
    int tmp;
    if(rand2 < rand1)
    {
        tmp = rand1;
        rand1 = rand2;
        rand2 = tmp;
    }
    char offspring1[dim], offspring2[dim];
    for(int i=0; i<dim; i++)
    {
        offspring1[i] = p1[i];
        offspring2[i] = p2[i];
    }
    int pos1, pos2;
    char tmpChar;
    for(int i=rand1; i<=rand2; i++)
    {
        for(int j=0; j<dim; j++)
        {
            if(offspring1[j] == p2[i])
            {
                pos1 = j;
                break;
            }
        }
        tmpChar = offspring1[i];
        offspring1[i] = offspring1[pos1];
        offspring1[pos1] = tmpChar;

        for(int j=0; j<dim; j++)
        {
            if(offspring2[j] == p1[i])
            {
                pos2 = j;
                break;
            }
        }
        tmpChar = offspring2[i];
        offspring2[i] = offspring2[pos2];
        offspring2[pos2] = tmpChar;
    }
    for(int i=0; i<dim; i++)
    {
        ret[i] = offspring1[i];
        ret[dim+i] = offspring2[i];
    }
}

void gx(char* p1, char* p2, int dim, char* ret, double** dist)
{
    set<char> notIncluded1, notIncluded2;
    for(int i=0; i<dim; i++)
    {
        notIncluded1.insert(p1[i]);
        notIncluded2.insert(p1[i]);
    }
    ret[0] = p1[0];
    ret[dim] = p2[0];
    notIncluded1.erase(p1[0]);
    notIncluded2.erase(p2[0]);
    int p1i, p2i, p1j, p2j;
    double d1, d2;
    int retSize = 1;
    char curr1, curr2;
    while(retSize < dim)
    {
        curr1 = ret[retSize-1];
        curr2 = ret[dim + retSize-1];
        for(int i=0; i<dim; i++)
        {
            if(p1[i] == curr1)
            {
                p1i = i;
            }
            if(p1[i] == curr2)
            {
                p1j = i;
            }
        }
        for(int i=0; i<dim; i++)
        {
            if(p2[i] == curr1)
            {
                p2i = i;
            }
            if(p2[i] == curr2)
            {
                p2j = i;
            }
        }

        if((notIncluded1.find(p1[(p1i+1)%dim]) == notIncluded1.end()) || (notIncluded1.find(p2[(p2i+1)%dim]) == notIncluded1.end()) )
        {
            if( (notIncluded1.find(p1[(p1i+1)%dim]) != notIncluded1.end()) )
            {
                ret[retSize] = p1[(p1i+1)%dim];
                notIncluded1.erase(p1[(p1i+1)%dim]);
            }
            else if( (notIncluded1.find(p2[(p2i+1)%dim]) != notIncluded1.end()) )
            {
                ret[retSize] = p2[(p2i+1)%dim];
                notIncluded1.erase(p2[(p2i+1)%dim]);
            }
            else    //select a random node
            {
                set<char>::const_iterator it1(notIncluded1.begin());
                advance(it1,rand() % notIncluded1.size());
                char nextNode = *it1;
                ret[retSize] = nextNode;
                notIncluded1.erase(nextNode);
            }
        }
        else
        {
            d1 = dist[index(curr1)][index(p1[(p1i+1)%dim])];
            d2 = dist[index(curr1)][index(p2[(p2i+1)%dim])];
            if(d1 <= d2)
            {
                ret[retSize] = p1[(p1i+1)%dim];
                notIncluded1.erase(p1[(p1i+1)%dim]);
            }
            else
            {
                ret[retSize] = p2[(p2i+1)%dim];
                notIncluded1.erase(p2[(p2i+1)%dim]);
            }
        }

        if((notIncluded2.find(p1[(p1j+1)%dim]) == notIncluded2.end()) || (notIncluded2.find(p2[(p2j+1)%dim]) == notIncluded2.end()) )
        {
            if( (notIncluded2.find(p1[(p1j+1)%dim]) != notIncluded2.end()) )
            {
                ret[dim + retSize] = p1[(p1j+1)%dim];
                notIncluded2.erase(p1[(p1j+1)%dim]);
            }
            else if( (notIncluded2.find(p2[(p2j+1)%dim]) != notIncluded2.end()) )
            {
                ret[dim + retSize] = p2[(p2j+1)%dim];
                notIncluded2.erase(p2[(p2j+1)%dim]);
            }
            else    //select a random node
            {
                set<char>::const_iterator it2(notIncluded2.begin());
                advance(it2,rand() % notIncluded2.size());
                char nextNode = *it2;
                ret[dim + retSize] = nextNode;
                notIncluded2.erase(nextNode);
            }
        }
        else
        {
            d1 = dist[index(curr2)][index(p1[(p1j+1)%dim])];
            d2 = dist[index(curr2)][index(p2[(p2j+1)%dim])];
            if(d1 <= d2)
            {
                ret[dim + retSize] = p1[(p1j+1)%dim];
                notIncluded2.erase(p1[(p1j+1)%dim]);
            }
            else
            {
                ret[dim + retSize] = p2[(p2j+1)%dim];
                notIncluded2.erase(p2[(p2j+1)%dim]);
            }
        }

        retSize++;
    }
}

int exists(int cycleID[], int val, int dim)
{
    int i;
    for(i=0; i<dim; i++)
    {
        if(cycleID[i] == val)
        {
            return i;
        }
    }
    return i;
}

void cx(char* p1, char* p2, int dim, char* ret)
{
    int cycleID[dim];
    for(int i=0; i<dim; i++)
    {
        cycleID[i] = 0;
    }
    int begin = 0;
    int curr = 0;
    cycleID[0] = 1;
    int i;
    int currCycle = 1;
    int nextIndex = 0;
    while(nextIndex != dim)
    {
        while(p1[begin] != p2[curr])
        {
            for(i=0; i<dim; i++)
            {
                if(p1[i] == p2[curr])
                {
                    cycleID[i] = currCycle;
                    curr = i;
                    break;
                }
            }
        }
        currCycle++;
        nextIndex = exists(cycleID, 0, dim);
        begin = nextIndex;
        curr = nextIndex;
        cycleID[begin] = currCycle;
    }

    for(i=0; i<dim; i++)
    {
        if(cycleID[i]%2 == 0)
        {
            ret[i] = p2[i];
            ret[dim+i] = p1[i];
        }
        else
        {
            ret[i] = p1[i];
            ret[dim+i] = p2[i];
        }
    }
}

void erx(char* p1, char* p2, int dim, char* ret)
{
    map<char, string> neighbours; 
    map<char,string>::iterator it ,it1;
    size_t found;
    for(int i=0; i<dim; i++)
    {
        string s = "~";
        s.insert(0, 1, p1[(i-1+dim)%dim]);
        s.insert(0, 1, p1[(i+1)%dim]);
        int j=0;
        for(j=0; j<dim; j++)
        {
            if(p2[j] == p1[i])
                break;
        }
        found = s.find(p2[(j-1+dim)%dim]);
        if(found==string::npos)
            s.insert(0, 1, p2[(j-1+dim)%dim]);
        found = s.find(p2[(j+1)%dim]);
        if(found==string::npos)
            s.insert(0, 1, p2[(j+1)%dim]);
        neighbours[p1[i]] = s;
    }

    it = neighbours.begin();
    advance(it, rand() % neighbours.size());
    ret[0] = it->first;
    neighbours.erase(it);
    for(it=neighbours.begin(); it!=neighbours.end(); ++it)
    {
        found = (it->second).find(ret[0]);
        if(found!=string::npos)
            (it->second).erase(found);
    }
    int minSize;
    int retSize = 1;
    while(neighbours.size()>0)
    {
        minSize = dim;
        for(it=neighbours.begin(); it!=neighbours.end(); ++it)
        {
            if((it->second).size() < minSize)
            {
                minSize = (it->second).size();
                it1 = it;
            }
        }
        ret[retSize] = (it1->first);
        neighbours.erase(it1);
        for(it=neighbours.begin(); it!=neighbours.end(); ++it)
        {
            found = (it->second).find(ret[retSize]);
            if(found!=string::npos)
                (it->second).erase(found);
        }
        retSize++;
    }

}

//global variables
map<char, string> trueID;
map<char, double> nodesX;
map<char, double> nodesY;

int main(int argc, char *argv[])
{
    double start = omp_get_wtime();
    srand ( unsigned ( time(0) ) );
    int NUM_THREADS = 4;
    NUM_THREADS = atoi(argv[2]);

    //scan input
    ifstream inFile;
    inFile.open(argv[1]);
    string s;
    char currID;
    getline(inFile, s);
    int space = s.find_first_of(" ");
    if(s[space+1] == ':')
        space = space + 2;
    string outFileName = "output_" + s.substr(space+1) + ".txt";
    getline(inFile, s);
    int pos = s.find_first_of(" ") + 1;
    int dim = atoi(s.substr(pos).c_str());
    char initString[dim];
    getline(inFile, s);
    for(int i=0; i<dim; i++)
    {
        if(i>25)
            currID = (i - 26) + '0';
        else
        {
            currID = (char)(97 + i);
        }
        initString[i] = currID;
        getline(inFile,s);
        int start = 0;
        if(s[0] == ' ')
            start = 1;
        int end = s.find_first_of(" ", start);
        trueID[currID] = s.substr(start, end);
        start = end + 1;
        end = s.find_first_of(" ", start);
        nodesX[currID] = atof(s.substr(start, end).c_str());
        start = end + 1;
        nodesY[currID] = atof(s.substr(start).c_str());
    }
    inFile.close();
    
    //generate dist
    double **dist;
    dist = new double*[dim];
    for(int i=0; i<dim; i++)
    {
        dist[i] = new double[dim];
    }

    int i, j;
    #pragma omp parallel for num_threads(NUM_THREADS) \
    private(j) \
    shared(i, dim, dist, nodesX, nodesY) \
    schedule(static)
    for(i=0; i<dim; i++)
    {
        for(j=0; j<dim; j++)
        {
            dist[index(initString[i])][index(initString[j])] = distance(nodesX[initString[i]], nodesY[initString[i]], nodesX[initString[j]], nodesY[initString[j]]);
        }
    }

    //generate initial population
    int initSize = 500 + abs(dim-10)*380;    //considering dim >=10
    if(dim < 10)
        initSize = dim*dim;
    char** population;
    population = new char*[initSize];
    for(int i=0; i<initSize; i++)
    {
       population[i] = new char[dim];
    }
    double populationFitness[initSize];
    char fittest[dim];
    double fittestFitness = 0;
    double currPathLength = 0;

    #pragma omp parallel for num_threads(NUM_THREADS) \
    private(j) \
    shared(i, initSize, dim, population, populationFitness, dist) \
    schedule(static)
    for(i=0; i<initSize; i++)
    {
        string currPerm(initString, dim);
        random_shuffle(currPerm.begin(), currPerm.end());
        for(j=0; j<dim; j++)
        {
            population[i][j] = currPerm[j];
        }
        populationFitness[i] = evaluateFitness(population[i], dim, dist);
    }

    //not to be parallelized
    for(i=0; i<initSize; i++)
    {
        if(populationFitness[i] > fittestFitness)
        {
                strcopy(fittest, population[i], dim);
                fittestFitness = populationFitness[i];
        }
    }

    //allocate space for variables
    int numSelect = 2*initSize;
    double totalFitness = 0;
    double rndNumber, offset = 0;

    double offspringsFitness[numSelect];
    double roulleteProb[initSize];
    char **offsprings;
    char **parents;
    offsprings = new char*[numSelect];
    parents = new char*[numSelect];
    for(i=0; i<numSelect; i++)
    {
        offsprings[i] = new char[dim];
    }
    for(i=0; i<numSelect; i++)
    {
        parents[i] = new char[dim];
    }
    double **offspringsFitnessThread;
    char ***offspringsThread;
    offspringsFitnessThread = new double*[NUM_THREADS];
    offspringsThread = new char**[NUM_THREADS];
    for(int i=0; i<NUM_THREADS; i++)
    {
        offspringsFitnessThread[i] = new double[numSelect];
        offspringsThread[i] = new char*[numSelect];
        for(int j=0; j<numSelect; j++)
            offspringsThread[i][j] = new char[dim];
    }
    int numOffspringsThread[NUM_THREADS];

    int rand1, rand2;
    int numOffsprings;
    double maxFitness = fittestFitness;

    //main loop
    for(i=0; i<NUM_ITER; i++)
    {
        if(i%100 == 0)
            cout<<"ITER: "<<i<<endl;
        numOffsprings = 0;
        for(j=0; j<NUM_THREADS; j++)
            numOffspringsThread[j] = 0;

        //roullete wheel selection
        bool done;
        int select;
        double newProb;
        #pragma omp parallel for num_threads(NUM_THREADS) \
        private(done, select, rndNumber, newProb) \
        shared(j, numSelect, parents, population, dim) \
        schedule(guided)
        for(j=0; j<numSelect; j++)
        {
            done = false;
            while(!done)
            {
                select = rand() % initSize;
                newProb = populationFitness[select]/maxFitness;
                rndNumber = rand() / (double) RAND_MAX;
                if(rndNumber <= newProb)
                {
                    strcopy(parents[j], population[select], dim);
                    done = true;
                }
            }
        }

        //crossover
        int privIndex, tid;
        #pragma omp parallel for num_threads(NUM_THREADS) \
        private(rndNumber, tid) \
        shared(j, numSelect, parents, dist, dim, numOffspringsThread, offspringsThread, offspringsFitnessThread) \
        schedule(guided)
        for(j=0; j<numSelect-1; j+=2)
        {
            tid = omp_get_thread_num();
            rndNumber = rand() / (double) RAND_MAX;
            if(rndNumber <= CROSS_PROB)
            {
                if(rndNumber <= CX_PROB)    //do cx
                {
                    char cxOffsprings[2*dim];
                    cx(parents[j], parents[j+1], dim, cxOffsprings);
                    for(int k=0; k<dim; k++)
                    {
                        offspringsThread[tid][numOffspringsThread[tid]][k] = cxOffsprings[k];
                        offspringsThread[tid][numOffspringsThread[tid]+1][k] = cxOffsprings[k+dim];
                    }
                    offspringsFitnessThread[tid][numOffspringsThread[tid]] = evaluateFitness(offspringsThread[tid][numOffspringsThread[tid]], dim, dist);
                    offspringsFitnessThread[tid][numOffspringsThread[tid]+1] = evaluateFitness(offspringsThread[tid][numOffspringsThread[tid]+1], dim, dist);
                    numOffspringsThread[tid] += 2;
                }
                else if(rndNumber <= CX_PROB + PMX_PROB)    //do pmx
                {
                    char pmxOffsprings[2*dim];
                    pmx(parents[j], parents[j+1], dim, pmxOffsprings);
                    for(int k=0; k<dim; k++)
                    {
                        offspringsThread[tid][numOffspringsThread[tid]][k] = pmxOffsprings[k];
                        offspringsThread[tid][numOffspringsThread[tid]+1][k] = pmxOffsprings[k+dim];
                    }
                    offspringsFitnessThread[tid][numOffspringsThread[tid]] = evaluateFitness(offspringsThread[tid][numOffspringsThread[tid]], dim, dist);
                    offspringsFitnessThread[tid][numOffspringsThread[tid]+1] = evaluateFitness(offspringsThread[tid][numOffspringsThread[tid]+1], dim, dist);
                    numOffspringsThread[tid] += 2;
                }
                else if(rndNumber <= CX_PROB + PMX_PROB + ERX_PROB) //do erx
                {
                    char erxOffsprings[dim];
                    erx(parents[j], parents[j+1], dim, erxOffsprings);
                    strcopy(offspringsThread[tid][numOffspringsThread[tid]], erxOffsprings, dim);
                    offspringsFitnessThread[tid][numOffspringsThread[tid]] = evaluateFitness(offspringsThread[tid][numOffspringsThread[tid]], dim, dist);                 
                    numOffspringsThread[tid]++;
                }
                else    //do gx
                {
                    char gxOffsprings[2*dim];
                    gx(parents[j], parents[j+1], dim, gxOffsprings, dist);
                    for(int k=0; k<dim; k++)
                    {
                        offspringsThread[tid][numOffspringsThread[tid]][k] = gxOffsprings[k];
                        offspringsThread[tid][numOffspringsThread[tid]+1][k] = gxOffsprings[k+dim];
                    }
                    offspringsFitnessThread[tid][numOffspringsThread[tid]] = evaluateFitness(offspringsThread[tid][numOffspringsThread[tid]], dim, dist);
                    offspringsFitnessThread[tid][numOffspringsThread[tid]+1] = evaluateFitness(offspringsThread[tid][numOffspringsThread[tid]+1], dim, dist);
                    numOffspringsThread[tid] += 2;
                }
                
            }
            else
            {
                for(int k=0; k<dim; k++)
                {
                    offspringsThread[tid][numOffspringsThread[tid]][k] = parents[j][k];
                    offspringsThread[tid][numOffspringsThread[tid]+1][k] = parents[j+1][k];
                }
                offspringsFitnessThread[tid][numOffspringsThread[tid]] = evaluateFitness(offspringsThread[tid][numOffspringsThread[tid]], dim, dist);
                offspringsFitnessThread[tid][numOffspringsThread[tid]+1] = evaluateFitness(offspringsThread[tid][numOffspringsThread[tid]+1], dim, dist);
                numOffspringsThread[tid] += 2;
            }
        }

        //combining results of crossover
        for(j=0; j<NUM_THREADS; j++)
        {
            for(int k=0; k<numOffspringsThread[j]; k++)
            {
                strcopy(offsprings[numOffsprings], offspringsThread[j][k], dim);
                offspringsFitness[numOffsprings] = offspringsFitnessThread[j][k];
                numOffsprings++;
            }
        }

        priority_queue<double> pq;
        //not to be parallelized
        for(int j=0; j<numOffsprings; j++)
        {
            pq.push(offspringsFitness[j]);
            //select best offspring
            if(offspringsFitness[j] > fittestFitness)
            {
                strcopy(fittest, offsprings[j], dim);
                fittestFitness = offspringsFitness[j];
            }
        }
        for(int j=0; j<initSize; j++)
        {
            pq.pop();
        }
        double threshold = 0;
        if(!pq.empty())
            threshold = pq.top();

        int newPopSize = 0;

        //select best set of offsprings
        //cannot be parallelized since need a break statement
        for(int j=0; j<numOffsprings; j++)
        {
            if(offspringsFitness[j] > threshold)
            {
                strcopy(population[newPopSize], offsprings[j], dim);
                populationFitness[newPopSize] = offspringsFitness[j];
                newPopSize++;
                if(newPopSize == initSize)
                    break;
            }
        }

        //mutation
        maxFitness = 0;
        char tmpChar;
        #pragma omp parallel for num_threads(NUM_THREADS) \
        private(rndNumber, rand1, rand2, tmpChar) \
        shared(j ,initSize, dim, population, populationFitness, dist) \
        schedule(guided)
        for(int j=0; j<initSize; j++)
        {
            rndNumber = rand() / (double) RAND_MAX;
            if(rndNumber <= MUT_PROB)
            {
                rand1 = rand() % dim;
                rand2 = rand() % dim;
                while(rand2 == rand1)
                {
                    rand2 = rand() % dim;
                }
                tmpChar = population[j][rand1];
                population[j][rand1] = population[j][rand2];
                population[j][rand2] = tmpChar;
            }
            populationFitness[j] = evaluateFitness(population[j], dim, dist);
        }

        //select best offspring
        //not to be parallelized
        for(int j=0; j<initSize; j++)
        {
            if(populationFitness[j] > fittestFitness)
            {
                strcopy(fittest, population[j], dim);
                fittestFitness = populationFitness[j];
            }
            if(populationFitness[j] > maxFitness)
            {
                maxFitness = populationFitness[j];
            }
        }

    }
    double end = omp_get_wtime();
    cout<<"Shortest path: "<<1/fittestFitness<<endl;
    cout<<"Time: "<<end-start<<endl;

    //writing to file
    ofstream outFile;
    outFile.open(outFileName.c_str());
    outFile<<"DIMENSION : "<<dim<<endl;
    outFile<<"TOUR_LENGTH : "<<1/fittestFitness<<endl;
    outFile<<"TOUR_SECTION"<<endl;
    for(i=0; i<dim; i++)
    {
        outFile<<trueID[fittest[i]]<<endl;
    }
    outFile<<-1;
    outFile.close();
    return 0;
}

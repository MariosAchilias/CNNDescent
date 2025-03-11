%Module(name=cnndescent, language="C++")

enum Dist {
    EUCLIDEAN,
    MANHATTAN
};

class CNNIndex {
    %TypeHeaderCode
    #include <cnnindex.hpp>
    %End
public:
    int k;
    int dim;
    CNNIndex(int k, int dim, Dist dist);
    ~CNNIndex();
    void set_n_threads(int n);
    void build_index_bruteforce();
    void build_index_nndescent(float precision, float sample_rate, int n_trees);
    void add_point(SIP_PYLIST);
%MethodCode
    float *tmp = (float *) malloc(sipCpp->dim * sizeof(float));
    for (int i = 0; i < sipCpp->dim; i++) {
        tmp[i] = PyFloat_AsDouble(PyList_GetItem(a0, i));
    }
    sipCpp->add_point(tmp);
    free(tmp);
%End
    SIP_PYLIST get_k_nearest(SIP_PYLIST);
%MethodCode
    float *point = (float *) malloc(sipCpp->dim * sizeof(float));
    for (int i = 0; i < sipCpp->dim; i++) {
        point[i] = PyFloat_AsDouble(PyList_GetItem(a0, i));
    }
    Neighbor *knn = KNNGraph_KNearest(sipCpp->graph, point);
    free(point);

    PyObject *knn_ = PyList_New(sipCpp->k);
    for (int i = 0; i < sipCpp->k; i++) {
        PyObject *tup = Py_BuildValue("(i,f)", knn[i].id, knn[i].dist);
        PyList_SetItem(knn_, i, tup);
    }
    free(knn);
    return knn_;
%End

    
};
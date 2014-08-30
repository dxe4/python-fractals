#include <Python.h>
#include <gmp.h>

PyObject* choose2(unsigned char x[], unsigned char y[]) {
    
    mpf_t number;
    mpf_init2(number, 128);
    mpf_set_str(number, "0.0415373652931074065807663354", 10);

    char* buffer;
    PyObject* result;
    buffer = malloc(mpz_sizeinbase(number, 10) + 2);
    // mpz_get_str(buffer, 10, mi_result);
    result = PyString_FromString(buffer);

    mpz_clear(number);
    free(buffer);
    Py_DECREF(result);

    return result;
}

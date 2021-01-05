# Use the compiler options and link commands in file:///opt/intel/documentation_2019/en/mkl/common/mkl_link_line_advisor.htm .
# The LAPACK variables can be defined in shell by running the following command.
#	source /opt/intel/compilers_and_libraries_2019/mac/bin/compilervars.sh intel64
# See also for conditional statements in Makefile: https://www.gnu.org/software/make/manual/html_node/Conditional-Syntax.html
MODE=RUN
ifdef db
	MODE=DEBUG
endif
ifeq ($(MODE), DEBUG)
	CC = gcc
	OPTS = -O${db} -g
	REPORT = $()
	TARGET = convert
	LDFLAGS = $()
	LIBS_MATH = -lm
else
	CC = icc
	OPTS = -O3
	REPORT = -qopt-report-phase=vec -qopt-report=5
	TARGET = convert.so
	LDFLAGS = -shared
endif
### To do: the following 3 lines should not be apllicable for the debug mode.
#ifeq ($(strip $(shell icc --version)),)
#	CC = icc
#	OPTS = -O3
#endif

CFLAGS = -fPIC -Wall -Wextra -std=c11 $(OPTS) # $(REPORT)
RM = rm
SRC_DIR = cython

# Detecting the OS type.
OS := $(shell uname -s)
$(info Make is being run in ${MODE} mode on the ${OS} OS.)

ifeq ($(OS), Darwin)
	# CFLAGS_MKL = -I${MKLROOT}/include
	# LIBS_MKL = -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_rt -lpthread $(LIBS) -ldl
	MKL_DIR = ${MKLROOT}
	CFLAGS_MKL = -m64 -I${MKL_DIR}/include
	LIBS_MKL =  -L${MKL_DIR}/lib -Wl, -lmkl_rt -lpthread
else ifeq ($(OS), Linux)
	ifeq ($(MKLROOT),)
		MKLROOT="/mnt/c/Program Files (x86)/IntelSWTools/compilers_and_libraries_2020.4.311/windows/mkl/"
	endif
	MKL_DIR = ${MKLROOT}
	CFLAGS_MKL = -m64 -I${MKL_DIR}/include
	LIBS_MKL =  -L${MKL_DIR}/lib/intel64 -Wl,--no-as-needed -lmkl_rt -lpthread -ldl
else
	MKL_DIR = "c:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2020/windows"
	CFLAGS_MKL =  -I${MKL_DIR}/mkl/include
	LIBS_DIR = ${MKL_DIR}/mkl/lib/intel64_win
	LIBS_MKL = -L${LIBS_DIR}/mkl_rt.lib
endif

$(shell mkdir -p obj/)

$(TARGET):obj/main.o obj/pauli.o obj/tracedot.o obj/zdot.o obj/krauss_theta.o obj/krauss_ptm.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) obj/main.o obj/pauli.o obj/tracedot.o obj/krauss_theta.o obj/krauss_ptm.o $(LIBS_MKL) obj/zdot.o $(LIBS_MATH)

obj/main.o: $(SRC_DIR)/main.c Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/main.c -o obj/main.o $(LIBS_MATH)

obj/pauli.o: $(SRC_DIR)/pauli.c $(SRC_DIR)/pauli.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/pauli.c -o obj/pauli.o $(LIBS_MATH)

obj/tracedot.o: $(SRC_DIR)/tracedot.c $(SRC_DIR)/tracedot.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/tracedot.c -o obj/tracedot.o $(LIBS_MATH)

obj/zdot.o: $(SRC_DIR)/zdot.c $(SRC_DIR)/zdot.h Makefile
	$(CC) $(CFLAGS) $(CFLAGS_MKL) $(LIBS_MKL) -c $(SRC_DIR)/zdot.c -o obj/zdot.o $(LIBS_MATH)

obj/krauss_theta.o: $(SRC_DIR)/krauss_theta.c $(SRC_DIR)/krauss_theta.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/krauss_theta.c -o obj/krauss_theta.o $(LIBS_MATH)

obj/krauss_ptm.o: $(SRC_DIR)/krauss_ptm.c $(SRC_DIR)/krauss_ptm.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/krauss_ptm.c -o obj/krauss_ptm.o $(LIBS_MATH)

clean:
	$(RM) convert convert.so obj/main.o obj/pauli.o obj/tracedot.o obj/zdot.o obj/krauss_theta.o obj/krauss_ptm.o

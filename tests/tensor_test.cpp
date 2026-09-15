#include <gtest/gtest.h>

#include <Math/tensor.h>

#include <stdexcept>
#include <type_traits>
#include <vector>

using math::Tensor;

namespace {

// Fills a tensor with 1, 2, 3, ... in row-major order.
template <typename T>
void iota_fill(Tensor<T>& t) {
    const auto& shape = t.shape();
    T next = T(1);
    if (shape.size() == 1) {
        for (size_t i = 0; i < shape[0]; ++i) t(i) = next++;
    } else if (shape.size() == 2) {
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j) t(i, j) = next++;
    } else {
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                for (size_t k = 0; k < shape[2]; ++k) t(i, j, k) = next++;
    }
}

}  // namespace

// --- construction -----------------------------------------------------------

TEST(TensorConstruction, DefaultsToZero) {
    Tensor<float> t({2, 3});
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t(i, j), 0.0f) << "at (" << i << ", " << j << ")";
        }
    }
}

TEST(TensorConstruction, InitialValueFillsEveryElement) {
    Tensor<float> t({2, 3}, 4.5f);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t(i, j), 4.5f);
        }
    }
}

TEST(TensorConstruction, ShapeAndSizeReflectDimensions) {
    Tensor<float> t({2, 3, 4});
    EXPECT_EQ(t.shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_EQ(t.rank(), 3u);
    EXPECT_EQ(t.size(), 24u);
}

TEST(TensorConstruction, RejectsZeroDimension) {
    EXPECT_THROW(Tensor<float>({2, 0}), std::invalid_argument);
    EXPECT_THROW(Tensor<float>({0, 2}), std::invalid_argument);
}

TEST(TensorConstruction, SupportsRankOneAndThree) {
    Tensor<float> v({5}, 1.0f);
    EXPECT_EQ(v.size(), 5u);
    EXPECT_FLOAT_EQ(v(4), 1.0f);

    Tensor<float> c({2, 3, 4}, 2.0f);
    EXPECT_EQ(c.size(), 24u);
    EXPECT_FLOAT_EQ(c(1, 2, 3), 2.0f);
}

// --- element access ---------------------------------------------------------

TEST(TensorAccess, WritesAreReadBack) {
    Tensor<float> t({2, 3});
    t(0, 0) = 1.0f;
    t(1, 2) = 6.0f;

    EXPECT_FLOAT_EQ(t(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t(1, 2), 6.0f);
    EXPECT_FLOAT_EQ(t(0, 1), 0.0f);
}

TEST(TensorAccess, StorageIsRowMajor) {
    Tensor<float> t({2, 3});
    iota_fill(t);

    // Last dimension contiguous: (0,0)=1 .. (0,2)=3, (1,0)=4 .. (1,2)=6.
    EXPECT_FLOAT_EQ(t(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(t(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(t(1, 2), 6.0f);
}

TEST(TensorAccess, ConstOverloadReads) {
    Tensor<float> t({2, 2}, 3.0f);
    const Tensor<float>& ct = t;

    EXPECT_FLOAT_EQ(ct(1, 1), 3.0f);
    static_assert(std::is_same_v<decltype(ct(0, 0)), const float&>,
                  "const operator() must yield a const reference");
}

TEST(TensorAccess, SpanOverloadMatchesVariadic) {
    Tensor<float> t({2, 3});
    iota_fill(t);

    std::vector<size_t> idx{1, 2};
    EXPECT_FLOAT_EQ(t(idx), t(1, 2));

    t(idx) = 99.0f;
    EXPECT_FLOAT_EQ(t(1, 2), 99.0f);
}

// --- fill -------------------------------------------------------------------

TEST(TensorFill, OverwritesPreviousContents) {
    Tensor<float> t({2, 3});
    iota_fill(t);
    t.fill(-1.0f);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t(i, j), -1.0f);
        }
    }
}

// --- compound assignment ----------------------------------------------------

TEST(TensorCompound, AddAssign) {
    Tensor<float> a({2, 2}, 1.0f);
    Tensor<float> b({2, 2}, 2.0f);
    a += b;
    EXPECT_FLOAT_EQ(a(0, 0), 3.0f);
    EXPECT_FLOAT_EQ(a(1, 1), 3.0f);
    EXPECT_FLOAT_EQ(b(0, 0), 2.0f) << "right operand must be untouched";
}

TEST(TensorCompound, SubtractAssign) {
    Tensor<float> a({2, 2}, 5.0f);
    Tensor<float> b({2, 2}, 2.0f);
    a -= b;
    EXPECT_FLOAT_EQ(a(0, 0), 3.0f);
}

TEST(TensorCompound, MultiplyAssignIsElementwise) {
    Tensor<float> a({2, 2}, 3.0f);
    Tensor<float> b({2, 2}, 4.0f);
    a *= b;
    EXPECT_FLOAT_EQ(a(0, 0), 12.0f);
}

TEST(TensorCompound, DivideAssign) {
    Tensor<float> a({2, 2}, 8.0f);
    Tensor<float> b({2, 2}, 2.0f);
    a /= b;
    EXPECT_FLOAT_EQ(a(0, 0), 4.0f);
}

TEST(TensorCompound, ScalarMultiplyAndDivide) {
    Tensor<float> t({2, 2}, 3.0f);
    t *= 2.0f;
    EXPECT_FLOAT_EQ(t(0, 0), 6.0f);
    t /= 3.0f;
    EXPECT_FLOAT_EQ(t(0, 0), 2.0f);
}

TEST(TensorCompound, ReturnsReferenceNotCopy) {
    Tensor<float> a({2, 2}, 1.0f);
    Tensor<float> b({2, 2}, 1.0f);
    (a += b) += b;   // chaining must keep mutating `a`
    EXPECT_FLOAT_EQ(a(0, 0), 3.0f);
}

TEST(TensorCompound, ShapeMismatchThrows) {
    Tensor<float> a({2, 2}, 1.0f);
    Tensor<float> b({3, 3}, 1.0f);

    EXPECT_THROW(a += b, std::invalid_argument);
    EXPECT_THROW(a -= b, std::invalid_argument);
    EXPECT_THROW(a *= b, std::invalid_argument);
    EXPECT_THROW(a /= b, std::invalid_argument);
}

TEST(TensorCompound, DivisionByZeroThrows) {
    Tensor<float> a({2, 2}, 1.0f);
    Tensor<float> zeros({2, 2}, 0.0f);

    EXPECT_THROW(a /= zeros, std::invalid_argument);
    EXPECT_THROW(a /= 0.0f, std::invalid_argument);
}

// --- free operators ---------------------------------------------------------

TEST(TensorOperators, AddsWithoutMutatingOperands) {
    Tensor<float> a({2, 2}, 1.0f);
    Tensor<float> b({2, 2}, 2.0f);

    const Tensor<float> c = a + b;
    EXPECT_FLOAT_EQ(c(0, 0), 3.0f);
    EXPECT_FLOAT_EQ(a(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(b(0, 0), 2.0f);
}

TEST(TensorOperators, SubtractsElementwise) {
    Tensor<float> a({2, 2}, 5.0f);
    Tensor<float> b({2, 2}, 2.0f);
    EXPECT_FLOAT_EQ((a - b)(0, 0), 3.0f);
}

TEST(TensorOperators, MultipliesElementwise) {
    Tensor<float> a({2, 2}, 3.0f);
    Tensor<float> b({2, 2}, 4.0f);
    EXPECT_FLOAT_EQ((a * b)(0, 0), 12.0f);
}

TEST(TensorOperators, DividesElementwise) {
    Tensor<float> a({2, 2}, 8.0f);
    Tensor<float> b({2, 2}, 2.0f);
    EXPECT_FLOAT_EQ((a / b)(0, 0), 4.0f);
}

TEST(TensorOperators, ScalarOnTheRight) {
    Tensor<float> t({2, 2}, 3.0f);
    EXPECT_FLOAT_EQ((t * 2.0f)(0, 0), 6.0f);
    EXPECT_FLOAT_EQ((t / 3.0f)(0, 0), 1.0f);
}

TEST(TensorOperators, ScalarOnTheLeft) {
    Tensor<float> t({2, 2}, 3.0f);
    EXPECT_FLOAT_EQ((2.0f * t)(0, 0), 6.0f);
}

// --- matmul -----------------------------------------------------------------

TEST(TensorMatMul, KnownProduct) {
    // [1 2 3]   [1 2]   [22 28]
    // [4 5 6] x [3 4] = [49 64]
    //           [5 6]
    Tensor<float> a({2, 3});
    iota_fill(a);
    Tensor<float> b({3, 2});
    iota_fill(b);

    const Tensor<float> c = math::MatMul(a, b);

    EXPECT_EQ(c.shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(c(0, 0), 22.0f);
    EXPECT_FLOAT_EQ(c(0, 1), 28.0f);
    EXPECT_FLOAT_EQ(c(1, 0), 49.0f);
    EXPECT_FLOAT_EQ(c(1, 1), 64.0f);
}

TEST(TensorMatMul, IdentityIsNeutral) {
    Tensor<float> a({2, 2});
    iota_fill(a);

    Tensor<float> id({2, 2}, 0.0f);
    id(0, 0) = 1.0f;
    id(1, 1) = 1.0f;

    const Tensor<float> c = math::MatMul(a, id);
    EXPECT_FLOAT_EQ(c(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(c(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(c(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(c(1, 1), 4.0f);
}

TEST(TensorMatMul, IsNotCommutative) {
    Tensor<float> a({2, 2});
    iota_fill(a);
    Tensor<float> b({2, 2}, 2.0f);
    b(0, 1) = 5.0f;

    const Tensor<float> ab = math::MatMul(a, b);
    const Tensor<float> ba = math::MatMul(b, a);
    EXPECT_NE(ab(0, 1), ba(0, 1));
}

TEST(TensorMatMul, RejectsMismatchedInnerDimensions) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<float> b({2, 3}, 1.0f);   // inner dims 3 vs 2
    EXPECT_THROW(math::MatMul(a, b), std::invalid_argument);
}

TEST(TensorMatMul, RejectsNonMatrixOperands) {
    Tensor<float> vec({3}, 1.0f);
    Tensor<float> mat({3, 3}, 1.0f);
    EXPECT_THROW(math::MatMul(vec, mat), std::invalid_argument);
    EXPECT_THROW(math::MatMul(mat, vec), std::invalid_argument);
}

TEST(TensorMatMul, NonSquareShapesPropagate) {
    Tensor<float> a({4, 2}, 1.0f);
    Tensor<float> b({2, 7}, 1.0f);
    const Tensor<float> c = math::MatMul(a, b);

    EXPECT_EQ(c.shape(), (std::vector<size_t>{4, 7}));
    EXPECT_FLOAT_EQ(c(0, 0), 2.0f);   // sum over the shared dimension of 2
}

// --- broadcasting -----------------------------------------------------------

TEST(TensorBroadcast, BiasAddsAlongLastAxis) {
    Tensor<float> out({2, 3}, 1.0f);
    Tensor<float> bias({3});
    bias(0) = 10.0f;
    bias(1) = 20.0f;
    bias(2) = 30.0f;

    const Tensor<float> r = out + bias;

    EXPECT_EQ(r.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(r(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(r(0, 2), 31.0f);
    EXPECT_FLOAT_EQ(r(1, 0), 11.0f);
    EXPECT_FLOAT_EQ(r(1, 2), 31.0f);
}

TEST(TensorBroadcast, ColumnVectorBroadcastsAlongRows) {
    Tensor<float> a({2, 3}, 0.0f);
    Tensor<float> col({2, 1});
    col(0, 0) = 5.0f;
    col(1, 0) = 7.0f;

    const Tensor<float> r = a + col;

    EXPECT_EQ(r.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(r(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(r(0, 2), 5.0f);
    EXPECT_FLOAT_EQ(r(1, 0), 7.0f);
    EXPECT_FLOAT_EQ(r(1, 2), 7.0f);
}

TEST(TensorBroadcast, BothOperandsExpand) {
    Tensor<float> row({1, 3});
    iota_fill(row);
    Tensor<float> col({2, 1});
    col(0, 0) = 10.0f;
    col(1, 0) = 20.0f;

    const Tensor<float> r = row + col;

    EXPECT_EQ(r.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(r(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(r(0, 2), 13.0f);
    EXPECT_FLOAT_EQ(r(1, 0), 21.0f);
    EXPECT_FLOAT_EQ(r(1, 2), 23.0f);
}

TEST(TensorBroadcast, LowerRankPadsOnTheLeft) {
    Tensor<float> a({2, 2, 3}, 1.0f);
    Tensor<float> b({3}, 2.0f);

    const Tensor<float> r = a * b;

    EXPECT_EQ(r.shape(), (std::vector<size_t>{2, 2, 3}));
    EXPECT_FLOAT_EQ(r(0, 0, 0), 2.0f);
    EXPECT_FLOAT_EQ(r(1, 1, 2), 2.0f);
}

TEST(TensorBroadcast, SingleElementActsLikeAScalar) {
    Tensor<float> a({2, 3}, 4.0f);
    Tensor<float> one({1}, 2.0f);

    const Tensor<float> r = a / one;

    EXPECT_EQ(r.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(r(1, 2), 2.0f);
}

TEST(TensorBroadcast, IncompatibleShapesThrow) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<float> b({4}, 1.0f);
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(TensorBroadcast, AllOperatorsBroadcast) {
    Tensor<float> a({2, 2}, 6.0f);
    Tensor<float> b({2}, 2.0f);

    EXPECT_FLOAT_EQ((a + b)(1, 1), 8.0f);
    EXPECT_FLOAT_EQ((a - b)(1, 1), 4.0f);
    EXPECT_FLOAT_EQ((a * b)(1, 1), 12.0f);
    EXPECT_FLOAT_EQ((a / b)(1, 1), 3.0f);
}

TEST(TensorBroadcast, CompoundBroadcastsRightOperand) {
    Tensor<float> out({2, 3}, 1.0f);
    Tensor<float> bias({3}, 10.0f);

    out += bias;

    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(out(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(out(1, 2), 11.0f);
}

TEST(TensorBroadcast, CompoundRejectsGrowingTheLeftOperand) {
    Tensor<float> small({3}, 1.0f);
    Tensor<float> big({2, 3}, 1.0f);

    EXPECT_THROW(small += big, std::invalid_argument);
    EXPECT_THROW(small *= big, std::invalid_argument);
}

TEST(TensorBroadcast, OperandsAreNotMutated) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<float> b({3}, 2.0f);

    const Tensor<float> r = a + b;
    (void)r;

    EXPECT_FLOAT_EQ(a(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(b(0), 2.0f);
    EXPECT_EQ(b.shape(), (std::vector<size_t>{3}));
}

// --- storage sharing --------------------------------------------------------

TEST(TensorStorage, CopyIsShallow) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<float> b = a;

    EXPECT_TRUE(a.shares_storage(b));
    EXPECT_EQ(a.data(), b.data());

    b(0, 0) = 99.0f;
    EXPECT_FLOAT_EQ(a(0, 0), 99.0f) << "shallow copy must alias the original";
}

TEST(TensorStorage, CloneIsDeep) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<float> b = a.clone();

    EXPECT_FALSE(a.shares_storage(b));
    EXPECT_NE(a.data(), b.data());

    b(0, 0) = 99.0f;
    EXPECT_FLOAT_EQ(a(0, 0), 1.0f) << "clone must not alias the original";
    EXPECT_FLOAT_EQ(b(0, 0), 99.0f);
}

TEST(TensorStorage, ClonePreservesShapeAndContents) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> b = a.clone();

    EXPECT_EQ(b.shape(), a.shape());
    EXPECT_EQ(b.strides(), a.strides());
    EXPECT_EQ(b.size(), a.size());
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(b(i, j), a(i, j));
        }
    }
}

TEST(TensorStorage, UseCountTracksSharers) {
    Tensor<float> a({2, 2}, 1.0f);
    EXPECT_EQ(a.use_count(), 1);
    {
        Tensor<float> b = a;
        EXPECT_EQ(a.use_count(), 2);
        Tensor<float> c = a.clone();
        EXPECT_EQ(a.use_count(), 2);
        EXPECT_EQ(c.use_count(), 1);
    }
    EXPECT_EQ(a.use_count(), 1);
}

TEST(TensorStorage, DefaultConstructedHasNoBuffer) {
    Tensor<float> t;
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.data(), nullptr);
    EXPECT_EQ(t.use_count(), 0);
    EXPECT_NO_THROW(t.fill(1.0f));

    const Tensor<float> c = t.clone();
    EXPECT_EQ(c.size(), 0u);
}

TEST(TensorStorage, ScalarOperatorsDoNotAliasOperand) {
    Tensor<float> a({2, 2}, 3.0f);

    const Tensor<float> doubled = a * 2.0f;
    EXPECT_FLOAT_EQ(doubled(0, 0), 6.0f);
    EXPECT_FLOAT_EQ(a(0, 0), 3.0f) << "operand must be untouched";

    const Tensor<float> left = 2.0f * a;
    EXPECT_FLOAT_EQ(left(0, 0), 6.0f);
    EXPECT_FLOAT_EQ(a(0, 0), 3.0f);

    const Tensor<float> halved = a / 3.0f;
    EXPECT_FLOAT_EQ(halved(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(a(0, 0), 3.0f);
}

TEST(TensorStorage, ElementwiseOperatorsProduceFreshStorage) {
    Tensor<float> a({2, 2}, 1.0f);
    Tensor<float> b({2, 2}, 2.0f);

    const Tensor<float> c = a + b;
    EXPECT_FALSE(c.shares_storage(a));
    EXPECT_FALSE(c.shares_storage(b));
}

TEST(TensorStorage, TransposeProducesFreshStorage) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> t = a.transpose();
    EXPECT_FALSE(t.shares_storage(a));
    EXPECT_FLOAT_EQ(t(2, 1), a(1, 2));
}

// --- element type -----------------------------------------------------------

TEST(TensorElementType, WorksWithDouble) {
    Tensor<double> t({2, 2}, 0.5);
    t *= 4.0;
    EXPECT_DOUBLE_EQ(t(0, 0), 2.0);
}

TEST(TensorElementType, WorksWithInt) {
    Tensor<int> t({2, 2}, 3);
    t *= 2;
    EXPECT_EQ(t(1, 1), 6);
}

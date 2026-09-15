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

TEST(TensorStorage, TransposeSharesStorage) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> t = a.transpose();
    EXPECT_TRUE(t.shares_storage(a));
    EXPECT_FLOAT_EQ(t(2, 1), a(1, 2));
}

// --- transpose / permute ----------------------------------------------------

TEST(TensorTranspose, TwoDimensionalSwapsRowsAndColumns) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> t = a.transpose();

    EXPECT_EQ(t.shape(), (std::vector<size_t>{3, 2}));
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t(j, i), a(i, j)) << "at (" << i << ", " << j << ")";
        }
    }
}

TEST(TensorTranspose, IsItsOwnInverseInTwoDimensions) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> back = a.transpose().transpose();

    EXPECT_EQ(back.shape(), a.shape());
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(back(i, j), a(i, j));
        }
    }
}

TEST(TensorTranspose, NoArgumentReversesAllAxes) {
    Tensor<float> a({2, 3, 4});
    iota_fill(a);

    const Tensor<float> t = a.transpose();

    EXPECT_EQ(t.shape(), (std::vector<size_t>{4, 3, 2}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(t(k, j, i), a(i, j, k));
}

TEST(TensorTranspose, SwapsTwoNamedAxes) {
    Tensor<float> a({2, 3, 4});
    iota_fill(a);

    const Tensor<float> t = a.transpose(0, 2);

    EXPECT_EQ(t.shape(), (std::vector<size_t>{4, 3, 2}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(t(k, j, i), a(i, j, k));
}

TEST(TensorTranspose, SwapsInnerAxesLeavingBatchAlone) {
    Tensor<float> a({2, 3, 4});
    iota_fill(a);

    const Tensor<float> t = a.transpose(1, 2);

    EXPECT_EQ(t.shape(), (std::vector<size_t>{2, 4, 3}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(t(i, k, j), a(i, j, k));
}

TEST(TensorTranspose, SwappingAnAxisWithItselfIsIdentity) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> t = a.transpose(1, 1);

    EXPECT_EQ(t.shape(), a.shape());
    EXPECT_FLOAT_EQ(t(1, 2), a(1, 2));
}

TEST(TensorTranspose, RejectsOutOfRangeDimension) {
    Tensor<float> a({2, 3}, 1.0f);
    EXPECT_THROW(a.transpose(0, 2), std::invalid_argument);
    EXPECT_THROW(a.transpose(5, 0), std::invalid_argument);
}

TEST(TensorPermute, ArbitraryReordering) {
    Tensor<float> a({2, 3, 4});
    iota_fill(a);

    const Tensor<float> p = a.permute(2, 0, 1);

    EXPECT_EQ(p.shape(), (std::vector<size_t>{4, 2, 3}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(p(k, i, j), a(i, j, k));
}

TEST(TensorPermute, VectorAndVariadicFormsAgree) {
    Tensor<float> a({2, 3, 4});
    iota_fill(a);

    const Tensor<float> v = a.permute(std::vector<size_t>{1, 2, 0});
    const Tensor<float> d = a.permute(1, 2, 0);

    EXPECT_EQ(v.shape(), d.shape());
    EXPECT_FLOAT_EQ(v(2, 3, 1), d(2, 3, 1));
}

TEST(TensorPermute, IdentityPermutationIsAView) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> p = a.permute(0, 1);

    EXPECT_EQ(p.shape(), a.shape());
    EXPECT_TRUE(p.shares_storage(a));
    EXPECT_TRUE(p.is_contiguous());
    EXPECT_FLOAT_EQ(p(1, 2), a(1, 2));
}

TEST(TensorPermute, RejectsWrongLengthOrRepeatedAxes) {
    Tensor<float> a({2, 3, 4}, 1.0f);

    EXPECT_THROW(a.permute(0, 1), std::invalid_argument);
    EXPECT_THROW(a.permute(0, 1, 1), std::invalid_argument);
    EXPECT_THROW(a.permute(0, 1, 3), std::invalid_argument);
}

TEST(TensorPermute, ResultIsUsableInArithmetic) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> t = a.transpose();
    const Tensor<float> sum = t + t;

    EXPECT_EQ(sum.shape(), (std::vector<size_t>{3, 2}));
    EXPECT_FLOAT_EQ(sum(2, 1), a(1, 2) * 2.0f);
}

// --- views ------------------------------------------------------------------

TEST(TensorView, ReshapeSharesStorage) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> v = a.view(3, 2);

    EXPECT_TRUE(v.shares_storage(a));
    EXPECT_EQ(v.shape(), (std::vector<size_t>{3, 2}));
    EXPECT_EQ(v.size(), a.size());
    EXPECT_FLOAT_EQ(v(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(v(2, 1), 6.0f);
}

TEST(TensorView, WritesThroughReshapeAreVisible) {
    Tensor<float> a({2, 3}, 0.0f);
    Tensor<float> v = a.view(6);

    v(5) = 42.0f;
    EXPECT_FLOAT_EQ(a(1, 2), 42.0f);
}

TEST(TensorView, ReshapeRejectsElementCountChange) {
    Tensor<float> a({2, 3}, 1.0f);
    EXPECT_THROW(a.view(2, 2), std::invalid_argument);
    EXPECT_THROW(a.view(7), std::invalid_argument);
}

TEST(TensorView, ReshapeRejectsNonContiguousSource) {
    Tensor<float> a({2, 3}, 1.0f);
    const Tensor<float> t = a.transpose();

    EXPECT_FALSE(t.is_contiguous());
    EXPECT_THROW(t.view(6), std::invalid_argument);
    EXPECT_NO_THROW(t.clone().view(6));
}

TEST(TensorView, SliceSelectsASubRange) {
    Tensor<float> a({4, 3});
    iota_fill(a);

    const Tensor<float> s = a.slice(0, 1, 2);

    EXPECT_TRUE(s.shares_storage(a));
    EXPECT_EQ(s.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(s(0, 0), a(1, 0));
    EXPECT_FLOAT_EQ(s(1, 2), a(2, 2));
}

TEST(TensorView, SliceAlongColumnsIsNonContiguous) {
    Tensor<float> a({2, 4});
    iota_fill(a);

    const Tensor<float> s = a.slice(1, 1, 2);

    EXPECT_FALSE(s.is_contiguous());
    EXPECT_EQ(s.shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(s(0, 0), a(0, 1));
    EXPECT_FLOAT_EQ(s(1, 1), a(1, 2));
}

TEST(TensorView, WritesThroughSliceAreVisible) {
    Tensor<float> a({4, 3}, 0.0f);
    Tensor<float> s = a.slice(0, 2, 2);

    s.fill(9.0f);

    EXPECT_FLOAT_EQ(a(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(a(1, 2), 0.0f);
    EXPECT_FLOAT_EQ(a(2, 0), 9.0f);
    EXPECT_FLOAT_EQ(a(3, 2), 9.0f);
}

TEST(TensorView, SliceRejectsOutOfBounds) {
    Tensor<float> a({4, 3}, 1.0f);
    EXPECT_THROW(a.slice(2, 0, 1), std::invalid_argument);
    EXPECT_THROW(a.slice(0, 3, 2), std::invalid_argument);
    EXPECT_THROW(a.slice(0, 0, 0), std::invalid_argument);
}

TEST(TensorView, RowDropsTheFirstAxis) {
    Tensor<float> a({3, 4});
    iota_fill(a);

    const Tensor<float> r = a.row(1);

    EXPECT_TRUE(r.shares_storage(a));
    EXPECT_EQ(r.shape(), (std::vector<size_t>{4}));
    EXPECT_EQ(r.rank(), 1u);
    for (size_t j = 0; j < 4; ++j) {
        EXPECT_FLOAT_EQ(r(j), a(1, j));
    }
}

TEST(TensorView, RowOfRankThreeYieldsAMatrix) {
    Tensor<float> a({2, 3, 4});
    iota_fill(a);

    const Tensor<float> r = a.row(1);

    EXPECT_EQ(r.shape(), (std::vector<size_t>{3, 4}));
    EXPECT_TRUE(r.is_contiguous());
    EXPECT_FLOAT_EQ(r(2, 3), a(1, 2, 3));
}

TEST(TensorView, RowRejectsOutOfRangeIndex) {
    Tensor<float> a({3, 4}, 1.0f);
    EXPECT_THROW(a.row(3), std::invalid_argument);
}

TEST(TensorView, TransposeViewSwapsStridesWithoutCopying) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> t = a.transpose();

    EXPECT_TRUE(t.shares_storage(a));
    EXPECT_EQ(t.shape(), (std::vector<size_t>{3, 2}));
    EXPECT_EQ(t.strides(), (std::vector<size_t>{1, 3}));
    EXPECT_FALSE(t.is_contiguous());

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t(j, i), a(i, j));
        }
    }
}

TEST(TensorView, PermuteMatchesItsCompactedForm) {
    Tensor<float> a({2, 3, 4});
    iota_fill(a);

    const Tensor<float> v = a.permute(2, 0, 1);
    const Tensor<float> c = v.contiguous();

    EXPECT_EQ(v.shape(), c.shape());
    EXPECT_TRUE(v.shares_storage(a));
    EXPECT_FALSE(c.shares_storage(a));
    EXPECT_TRUE(c.is_contiguous());

    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 2; ++j)
            for (size_t k = 0; k < 3; ++k)
                EXPECT_FLOAT_EQ(v(i, j, k), c(i, j, k));
}

// --- operations over non-contiguous views -----------------------------------

TEST(TensorViewOps, CloneCompactsAView) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> t = a.transpose();
    const Tensor<float> c = t.clone();

    EXPECT_TRUE(c.is_contiguous());
    EXPECT_FALSE(c.shares_storage(a));
    EXPECT_EQ(c.size(), 6u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(c(i, j), t(i, j));
        }
    }
}

TEST(TensorViewOps, FillTouchesOnlyTheView) {
    Tensor<float> a({3, 3}, 1.0f);
    Tensor<float> middle = a.slice(1, 1, 1);

    middle.fill(7.0f);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(a(i, 0), 1.0f);
        EXPECT_FLOAT_EQ(a(i, 1), 7.0f);
        EXPECT_FLOAT_EQ(a(i, 2), 1.0f);
    }
}

TEST(TensorViewOps, ScalarMultiplyTouchesOnlyTheView) {
    Tensor<float> a({2, 4}, 2.0f);
    Tensor<float> half = a.slice(1, 0, 2);

    half *= 5.0f;

    EXPECT_FLOAT_EQ(a(0, 0), 10.0f);
    EXPECT_FLOAT_EQ(a(0, 1), 10.0f);
    EXPECT_FLOAT_EQ(a(0, 2), 2.0f);
    EXPECT_FLOAT_EQ(a(1, 3), 2.0f);
}

TEST(TensorViewOps, CompoundAssignWritesThroughAView) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<float> r = a.row(0);
    Tensor<float> bias({3}, 10.0f);

    r += bias;

    EXPECT_FLOAT_EQ(a(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(a(0, 2), 11.0f);
    EXPECT_FLOAT_EQ(a(1, 0), 1.0f);
}

TEST(TensorViewOps, CompoundAssignOnATransposedView) {
    Tensor<float> a({2, 3}, 0.0f);
    Tensor<float> t = a.transpose();
    Tensor<float> ones({3, 2}, 1.0f);

    t += ones;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a(i, j), 1.0f);
        }
    }
}

TEST(TensorViewOps, ArithmeticReadsNonContiguousOperands) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> t = a.transpose();
    const Tensor<float> sum = t + t;

    EXPECT_EQ(sum.shape(), (std::vector<size_t>{3, 2}));
    EXPECT_TRUE(sum.is_contiguous());
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(sum(i, j), t(i, j) * 2.0f);
        }
    }
}

TEST(TensorViewOps, MatMulAcceptsATransposedView) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> product = math::MatMul(a, a.transpose());

    EXPECT_EQ(product.shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(product(0, 0), 14.0f);
    EXPECT_FLOAT_EQ(product(0, 1), 32.0f);
    EXPECT_FLOAT_EQ(product(1, 0), 32.0f);
    EXPECT_FLOAT_EQ(product(1, 1), 77.0f);
}

TEST(TensorViewOps, ScalarOperatorOnAViewDoesNotTouchTheSource) {
    Tensor<float> a({2, 3}, 3.0f);
    const Tensor<float> r = a.row(0);

    const Tensor<float> doubled = r * 2.0f;

    EXPECT_FLOAT_EQ(doubled(0), 6.0f);
    EXPECT_FLOAT_EQ(a(0, 0), 3.0f);
    EXPECT_FALSE(doubled.shares_storage(a));
}

TEST(TensorViewOps, ContiguityReportedCorrectly) {
    Tensor<float> a({2, 3}, 1.0f);
    EXPECT_TRUE(a.is_contiguous());
    EXPECT_TRUE(a.row(0).is_contiguous());
    EXPECT_TRUE(a.slice(0, 0, 1).is_contiguous());
    EXPECT_FALSE(a.transpose().is_contiguous());
}

TEST(TensorViewOps, ViewOfAViewComposes) {
    Tensor<float> a({4, 4});
    iota_fill(a);

    const Tensor<float> block = a.slice(0, 1, 2).slice(1, 1, 2);

    EXPECT_EQ(block.shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(block(0, 0), a(1, 1));
    EXPECT_FLOAT_EQ(block(1, 1), a(2, 2));
}

// --- contiguous -------------------------------------------------------------

TEST(TensorContiguous, AlreadyContiguousSharesStorage) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> c = a.contiguous();

    EXPECT_TRUE(c.shares_storage(a));
    EXPECT_EQ(c.data(), a.data());
    EXPECT_TRUE(c.is_contiguous());
}

TEST(TensorContiguous, NonContiguousIsCompactedIntoNewStorage) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> t = a.transpose();
    ASSERT_FALSE(t.is_contiguous());

    const Tensor<float> c = t.contiguous();

    EXPECT_TRUE(c.is_contiguous());
    EXPECT_FALSE(c.shares_storage(a));
    EXPECT_EQ(c.shape(), t.shape());
    EXPECT_EQ(c.strides(), (std::vector<size_t>{2, 1}));
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(c(i, j), t(i, j));
        }
    }
}

TEST(TensorContiguous, UnblocksViewOnATransposedTensor) {
    Tensor<float> a({2, 3});
    iota_fill(a);

    const Tensor<float> t = a.transpose();
    EXPECT_THROW(t.view(6), std::invalid_argument);

    const Tensor<float> flat = t.contiguous().view(6);
    EXPECT_EQ(flat.shape(), (std::vector<size_t>{6}));
    EXPECT_FLOAT_EQ(flat(0), a(0, 0));
    EXPECT_FLOAT_EQ(flat(1), a(1, 0));
}

TEST(TensorContiguous, CompactedCopyIsDecoupledFromTheSource) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<float> c = a.transpose().contiguous();

    c.fill(9.0f);

    EXPECT_FLOAT_EQ(a(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(c(0, 0), 9.0f);
}

TEST(TensorContiguous, SliceOfRowsStaysContiguous) {
    Tensor<float> a({4, 3});
    iota_fill(a);

    const Tensor<float> s = a.slice(0, 1, 2);
    ASSERT_TRUE(s.is_contiguous());

    const Tensor<float> c = s.contiguous();
    EXPECT_TRUE(c.shares_storage(a)) << "no copy needed when already contiguous";
}

TEST(TensorContiguous, EmptyTensorIsHandled) {
    Tensor<float> t;
    const Tensor<float> c = t.contiguous();
    EXPECT_EQ(c.size(), 0u);
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

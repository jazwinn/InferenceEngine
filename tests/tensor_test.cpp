#include <gtest/gtest.h>

#include <Tensor/Tensor.h>

#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

using tensor::Tensor;

namespace {

// Fills a tensor with 1, 2, 3, ... in row-major order.
template <typename T>
void IotaFill(Tensor<T>& t) {
    const auto& shape = t.Shape();
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
    EXPECT_EQ(t.Shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_EQ(t.Rank(), 3u);
    EXPECT_EQ(t.Size(), 24u);
}

TEST(TensorConstruction, RejectsZeroDimension) {
    EXPECT_THROW(Tensor<float>({2, 0}), std::invalid_argument);
    EXPECT_THROW(Tensor<float>({0, 2}), std::invalid_argument);
}

TEST(TensorConstruction, SupportsRankOneAndThree) {
    Tensor<float> v({5}, 1.0f);
    EXPECT_EQ(v.Size(), 5u);
    EXPECT_FLOAT_EQ(v(4), 1.0f);

    Tensor<float> c({2, 3, 4}, 2.0f);
    EXPECT_EQ(c.Size(), 24u);
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
    IotaFill(t);

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
    IotaFill(t);

    std::vector<size_t> idx{1, 2};
    EXPECT_FLOAT_EQ(t(idx), t(1, 2));

    t(idx) = 99.0f;
    EXPECT_FLOAT_EQ(t(1, 2), 99.0f);
}

// --- fill -------------------------------------------------------------------

TEST(TensorFill, OverwritesPreviousContents) {
    Tensor<float> t({2, 3});
    IotaFill(t);
    t.Fill(-1.0f);

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
    IotaFill(a);
    Tensor<float> b({3, 2});
    IotaFill(b);

    const Tensor<float> c = tensor::MatMul(a, b);

    EXPECT_EQ(c.Shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(c(0, 0), 22.0f);
    EXPECT_FLOAT_EQ(c(0, 1), 28.0f);
    EXPECT_FLOAT_EQ(c(1, 0), 49.0f);
    EXPECT_FLOAT_EQ(c(1, 1), 64.0f);
}

TEST(TensorMatMul, IdentityIsNeutral) {
    Tensor<float> a({2, 2});
    IotaFill(a);

    Tensor<float> id({2, 2}, 0.0f);
    id(0, 0) = 1.0f;
    id(1, 1) = 1.0f;

    const Tensor<float> c = tensor::MatMul(a, id);
    EXPECT_FLOAT_EQ(c(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(c(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(c(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(c(1, 1), 4.0f);
}

TEST(TensorMatMul, IsNotCommutative) {
    Tensor<float> a({2, 2});
    IotaFill(a);
    Tensor<float> b({2, 2}, 2.0f);
    b(0, 1) = 5.0f;

    const Tensor<float> ab = tensor::MatMul(a, b);
    const Tensor<float> ba = tensor::MatMul(b, a);
    EXPECT_NE(ab(0, 1), ba(0, 1));
}

TEST(TensorMatMul, RejectsMismatchedInnerDimensions) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<float> b({2, 3}, 1.0f);   // inner dims 3 vs 2
    EXPECT_THROW(tensor::MatMul(a, b), std::invalid_argument);
}

TEST(TensorMatMul, RejectsNonMatrixOperands) {
    Tensor<float> vec({3}, 1.0f);
    Tensor<float> mat({3, 3}, 1.0f);
    EXPECT_THROW(tensor::MatMul(vec, mat), std::invalid_argument);
    EXPECT_THROW(tensor::MatMul(mat, vec), std::invalid_argument);
}

TEST(TensorMatMul, NonSquareShapesPropagate) {
    Tensor<float> a({4, 2}, 1.0f);
    Tensor<float> b({2, 7}, 1.0f);
    const Tensor<float> c = tensor::MatMul(a, b);

    EXPECT_EQ(c.Shape(), (std::vector<size_t>{4, 7}));
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

    EXPECT_EQ(r.Shape(), (std::vector<size_t>{2, 3}));
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

    EXPECT_EQ(r.Shape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(r(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(r(0, 2), 5.0f);
    EXPECT_FLOAT_EQ(r(1, 0), 7.0f);
    EXPECT_FLOAT_EQ(r(1, 2), 7.0f);
}

TEST(TensorBroadcast, BothOperandsExpand) {
    Tensor<float> row({1, 3});
    IotaFill(row);
    Tensor<float> col({2, 1});
    col(0, 0) = 10.0f;
    col(1, 0) = 20.0f;

    const Tensor<float> r = row + col;

    EXPECT_EQ(r.Shape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(r(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(r(0, 2), 13.0f);
    EXPECT_FLOAT_EQ(r(1, 0), 21.0f);
    EXPECT_FLOAT_EQ(r(1, 2), 23.0f);
}

TEST(TensorBroadcast, LowerRankPadsOnTheLeft) {
    Tensor<float> a({2, 2, 3}, 1.0f);
    Tensor<float> b({3}, 2.0f);

    const Tensor<float> r = a * b;

    EXPECT_EQ(r.Shape(), (std::vector<size_t>{2, 2, 3}));
    EXPECT_FLOAT_EQ(r(0, 0, 0), 2.0f);
    EXPECT_FLOAT_EQ(r(1, 1, 2), 2.0f);
}

TEST(TensorBroadcast, SingleElementActsLikeAScalar) {
    Tensor<float> a({2, 3}, 4.0f);
    Tensor<float> one({1}, 2.0f);

    const Tensor<float> r = a / one;

    EXPECT_EQ(r.Shape(), (std::vector<size_t>{2, 3}));
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

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{2, 3}));
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
    EXPECT_EQ(b.Shape(), (std::vector<size_t>{3}));
}

// --- storage sharing --------------------------------------------------------

TEST(TensorStorage, CopyIsShallow) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<float> b = a;

    EXPECT_TRUE(a.SharesStorage(b));
    EXPECT_EQ(a.Data(), b.Data());

    b(0, 0) = 99.0f;
    EXPECT_FLOAT_EQ(a(0, 0), 99.0f) << "shallow copy must alias the original";
}

TEST(TensorStorage, CloneIsDeep) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<float> b = a.Clone();

    EXPECT_FALSE(a.SharesStorage(b));
    EXPECT_NE(a.Data(), b.Data());

    b(0, 0) = 99.0f;
    EXPECT_FLOAT_EQ(a(0, 0), 1.0f) << "clone must not alias the original";
    EXPECT_FLOAT_EQ(b(0, 0), 99.0f);
}

TEST(TensorStorage, ClonePreservesShapeAndContents) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> b = a.Clone();

    EXPECT_EQ(b.Shape(), a.Shape());
    EXPECT_EQ(b.Strides(), a.Strides());
    EXPECT_EQ(b.Size(), a.Size());
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(b(i, j), a(i, j));
        }
    }
}

TEST(TensorStorage, UseCountTracksSharers) {
    Tensor<float> a({2, 2}, 1.0f);
    EXPECT_EQ(a.UseCount(), 1);
    {
        Tensor<float> b = a;
        EXPECT_EQ(a.UseCount(), 2);
        Tensor<float> c = a.Clone();
        EXPECT_EQ(a.UseCount(), 2);
        EXPECT_EQ(c.UseCount(), 1);
    }
    EXPECT_EQ(a.UseCount(), 1);
}

TEST(TensorStorage, DefaultConstructedHasNoBuffer) {
    Tensor<float> t;
    EXPECT_EQ(t.Size(), 0u);
    EXPECT_EQ(t.Data(), nullptr);
    EXPECT_EQ(t.UseCount(), 0);
    EXPECT_NO_THROW(t.Fill(1.0f));

    const Tensor<float> c = t.Clone();
    EXPECT_EQ(c.Size(), 0u);
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
    EXPECT_FALSE(c.SharesStorage(a));
    EXPECT_FALSE(c.SharesStorage(b));
}

TEST(TensorStorage, TransposeSharesStorage) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> t = a.Transpose();
    EXPECT_TRUE(t.SharesStorage(a));
    EXPECT_FLOAT_EQ(t(2, 1), a(1, 2));
}

// --- transpose / permute ----------------------------------------------------

TEST(TensorTranspose, TwoDimensionalSwapsRowsAndColumns) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> t = a.Transpose();

    EXPECT_EQ(t.Shape(), (std::vector<size_t>{3, 2}));
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t(j, i), a(i, j)) << "at (" << i << ", " << j << ")";
        }
    }
}

TEST(TensorTranspose, IsItsOwnInverseInTwoDimensions) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> back = a.Transpose().Transpose();

    EXPECT_EQ(back.Shape(), a.Shape());
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(back(i, j), a(i, j));
        }
    }
}

TEST(TensorTranspose, NoArgumentReversesAllAxes) {
    Tensor<float> a({2, 3, 4});
    IotaFill(a);

    const Tensor<float> t = a.Transpose();

    EXPECT_EQ(t.Shape(), (std::vector<size_t>{4, 3, 2}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(t(k, j, i), a(i, j, k));
}

TEST(TensorTranspose, SwapsTwoNamedAxes) {
    Tensor<float> a({2, 3, 4});
    IotaFill(a);

    const Tensor<float> t = a.Transpose(0, 2);

    EXPECT_EQ(t.Shape(), (std::vector<size_t>{4, 3, 2}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(t(k, j, i), a(i, j, k));
}

TEST(TensorTranspose, SwapsInnerAxesLeavingBatchAlone) {
    Tensor<float> a({2, 3, 4});
    IotaFill(a);

    const Tensor<float> t = a.Transpose(1, 2);

    EXPECT_EQ(t.Shape(), (std::vector<size_t>{2, 4, 3}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(t(i, k, j), a(i, j, k));
}

TEST(TensorTranspose, SwappingAnAxisWithItselfIsIdentity) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> t = a.Transpose(1, 1);

    EXPECT_EQ(t.Shape(), a.Shape());
    EXPECT_FLOAT_EQ(t(1, 2), a(1, 2));
}

TEST(TensorTranspose, RejectsOutOfRangeDimension) {
    Tensor<float> a({2, 3}, 1.0f);
    EXPECT_THROW(a.Transpose(0, 2), std::invalid_argument);
    EXPECT_THROW(a.Transpose(5, 0), std::invalid_argument);
}

TEST(TensorPermute, ArbitraryReordering) {
    Tensor<float> a({2, 3, 4});
    IotaFill(a);

    const Tensor<float> p = a.Permute(2, 0, 1);

    EXPECT_EQ(p.Shape(), (std::vector<size_t>{4, 2, 3}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(p(k, i, j), a(i, j, k));
}

TEST(TensorPermute, VectorAndVariadicFormsAgree) {
    Tensor<float> a({2, 3, 4});
    IotaFill(a);

    const Tensor<float> v = a.Permute(std::vector<size_t>{1, 2, 0});
    const Tensor<float> d = a.Permute(1, 2, 0);

    EXPECT_EQ(v.Shape(), d.Shape());
    EXPECT_FLOAT_EQ(v(2, 3, 1), d(2, 3, 1));
}

TEST(TensorPermute, IdentityPermutationIsAView) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> p = a.Permute(0, 1);

    EXPECT_EQ(p.Shape(), a.Shape());
    EXPECT_TRUE(p.SharesStorage(a));
    EXPECT_TRUE(p.IsContiguous());
    EXPECT_FLOAT_EQ(p(1, 2), a(1, 2));
}

TEST(TensorPermute, RejectsWrongLengthOrRepeatedAxes) {
    Tensor<float> a({2, 3, 4}, 1.0f);

    EXPECT_THROW(a.Permute(0, 1), std::invalid_argument);
    EXPECT_THROW(a.Permute(0, 1, 1), std::invalid_argument);
    EXPECT_THROW(a.Permute(0, 1, 3), std::invalid_argument);
}

TEST(TensorPermute, ResultIsUsableInArithmetic) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> t = a.Transpose();
    const Tensor<float> sum = t + t;

    EXPECT_EQ(sum.Shape(), (std::vector<size_t>{3, 2}));
    EXPECT_FLOAT_EQ(sum(2, 1), a(1, 2) * 2.0f);
}

// --- views ------------------------------------------------------------------

TEST(TensorView, ReshapeSharesStorage) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> v = a.View(3, 2);

    EXPECT_TRUE(v.SharesStorage(a));
    EXPECT_EQ(v.Shape(), (std::vector<size_t>{3, 2}));
    EXPECT_EQ(v.Size(), a.Size());
    EXPECT_FLOAT_EQ(v(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(v(2, 1), 6.0f);
}

TEST(TensorView, WritesThroughReshapeAreVisible) {
    Tensor<float> a({2, 3}, 0.0f);
    Tensor<float> v = a.View(6);

    v(5) = 42.0f;
    EXPECT_FLOAT_EQ(a(1, 2), 42.0f);
}

TEST(TensorView, ReshapeRejectsElementCountChange) {
    Tensor<float> a({2, 3}, 1.0f);
    EXPECT_THROW(a.View(2, 2), std::invalid_argument);
    EXPECT_THROW(a.View(7), std::invalid_argument);
}

TEST(TensorView, ReshapeRejectsNonContiguousSource) {
    Tensor<float> a({2, 3}, 1.0f);
    const Tensor<float> t = a.Transpose();

    EXPECT_FALSE(t.IsContiguous());
    EXPECT_THROW(t.View(6), std::invalid_argument);
    EXPECT_NO_THROW(t.Clone().View(6));
}

TEST(TensorView, SliceSelectsASubRange) {
    Tensor<float> a({4, 3});
    IotaFill(a);

    const Tensor<float> s = a.Slice(0, 1, 2);

    EXPECT_TRUE(s.SharesStorage(a));
    EXPECT_EQ(s.Shape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(s(0, 0), a(1, 0));
    EXPECT_FLOAT_EQ(s(1, 2), a(2, 2));
}

TEST(TensorView, SliceAlongColumnsIsNonContiguous) {
    Tensor<float> a({2, 4});
    IotaFill(a);

    const Tensor<float> s = a.Slice(1, 1, 2);

    EXPECT_FALSE(s.IsContiguous());
    EXPECT_EQ(s.Shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(s(0, 0), a(0, 1));
    EXPECT_FLOAT_EQ(s(1, 1), a(1, 2));
}

TEST(TensorView, WritesThroughSliceAreVisible) {
    Tensor<float> a({4, 3}, 0.0f);
    Tensor<float> s = a.Slice(0, 2, 2);

    s.Fill(9.0f);

    EXPECT_FLOAT_EQ(a(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(a(1, 2), 0.0f);
    EXPECT_FLOAT_EQ(a(2, 0), 9.0f);
    EXPECT_FLOAT_EQ(a(3, 2), 9.0f);
}

TEST(TensorView, SliceRejectsOutOfBounds) {
    Tensor<float> a({4, 3}, 1.0f);
    EXPECT_THROW(a.Slice(2, 0, 1), std::invalid_argument);
    EXPECT_THROW(a.Slice(0, 3, 2), std::invalid_argument);
    EXPECT_THROW(a.Slice(0, 0, 0), std::invalid_argument);
}

TEST(TensorView, RowDropsTheFirstAxis) {
    Tensor<float> a({3, 4});
    IotaFill(a);

    const Tensor<float> r = a.Row(1);

    EXPECT_TRUE(r.SharesStorage(a));
    EXPECT_EQ(r.Shape(), (std::vector<size_t>{4}));
    EXPECT_EQ(r.Rank(), 1u);
    for (size_t j = 0; j < 4; ++j) {
        EXPECT_FLOAT_EQ(r(j), a(1, j));
    }
}

TEST(TensorView, RowOfRankThreeYieldsAMatrix) {
    Tensor<float> a({2, 3, 4});
    IotaFill(a);

    const Tensor<float> r = a.Row(1);

    EXPECT_EQ(r.Shape(), (std::vector<size_t>{3, 4}));
    EXPECT_TRUE(r.IsContiguous());
    EXPECT_FLOAT_EQ(r(2, 3), a(1, 2, 3));
}

TEST(TensorView, RowRejectsOutOfRangeIndex) {
    Tensor<float> a({3, 4}, 1.0f);
    EXPECT_THROW(a.Row(3), std::invalid_argument);
}

TEST(TensorView, TransposeViewSwapsStridesWithoutCopying) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> t = a.Transpose();

    EXPECT_TRUE(t.SharesStorage(a));
    EXPECT_EQ(t.Shape(), (std::vector<size_t>{3, 2}));
    EXPECT_EQ(t.Strides(), (std::vector<size_t>{1, 3}));
    EXPECT_FALSE(t.IsContiguous());

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t(j, i), a(i, j));
        }
    }
}

TEST(TensorView, PermuteMatchesItsCompactedForm) {
    Tensor<float> a({2, 3, 4});
    IotaFill(a);

    const Tensor<float> v = a.Permute(2, 0, 1);
    const Tensor<float> c = v.Contiguous();

    EXPECT_EQ(v.Shape(), c.Shape());
    EXPECT_TRUE(v.SharesStorage(a));
    EXPECT_FALSE(c.SharesStorage(a));
    EXPECT_TRUE(c.IsContiguous());

    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 2; ++j)
            for (size_t k = 0; k < 3; ++k)
                EXPECT_FLOAT_EQ(v(i, j, k), c(i, j, k));
}

// --- operations over non-contiguous views -----------------------------------

TEST(TensorViewOps, CloneCompactsAView) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> t = a.Transpose();
    const Tensor<float> c = t.Clone();

    EXPECT_TRUE(c.IsContiguous());
    EXPECT_FALSE(c.SharesStorage(a));
    EXPECT_EQ(c.Size(), 6u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(c(i, j), t(i, j));
        }
    }
}

TEST(TensorViewOps, FillTouchesOnlyTheView) {
    Tensor<float> a({3, 3}, 1.0f);
    Tensor<float> middle = a.Slice(1, 1, 1);

    middle.Fill(7.0f);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(a(i, 0), 1.0f);
        EXPECT_FLOAT_EQ(a(i, 1), 7.0f);
        EXPECT_FLOAT_EQ(a(i, 2), 1.0f);
    }
}

TEST(TensorViewOps, ScalarMultiplyTouchesOnlyTheView) {
    Tensor<float> a({2, 4}, 2.0f);
    Tensor<float> half = a.Slice(1, 0, 2);

    half *= 5.0f;

    EXPECT_FLOAT_EQ(a(0, 0), 10.0f);
    EXPECT_FLOAT_EQ(a(0, 1), 10.0f);
    EXPECT_FLOAT_EQ(a(0, 2), 2.0f);
    EXPECT_FLOAT_EQ(a(1, 3), 2.0f);
}

TEST(TensorViewOps, CompoundAssignWritesThroughAView) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<float> r = a.Row(0);
    Tensor<float> bias({3}, 10.0f);

    r += bias;

    EXPECT_FLOAT_EQ(a(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(a(0, 2), 11.0f);
    EXPECT_FLOAT_EQ(a(1, 0), 1.0f);
}

TEST(TensorViewOps, CompoundAssignOnATransposedView) {
    Tensor<float> a({2, 3}, 0.0f);
    Tensor<float> t = a.Transpose();
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
    IotaFill(a);

    const Tensor<float> t = a.Transpose();
    const Tensor<float> sum = t + t;

    EXPECT_EQ(sum.Shape(), (std::vector<size_t>{3, 2}));
    EXPECT_TRUE(sum.IsContiguous());
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(sum(i, j), t(i, j) * 2.0f);
        }
    }
}

TEST(TensorViewOps, MatMulAcceptsATransposedView) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> product = tensor::MatMul(a, a.Transpose());

    EXPECT_EQ(product.Shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(product(0, 0), 14.0f);
    EXPECT_FLOAT_EQ(product(0, 1), 32.0f);
    EXPECT_FLOAT_EQ(product(1, 0), 32.0f);
    EXPECT_FLOAT_EQ(product(1, 1), 77.0f);
}

TEST(TensorViewOps, ScalarOperatorOnAViewDoesNotTouchTheSource) {
    Tensor<float> a({2, 3}, 3.0f);
    const Tensor<float> r = a.Row(0);

    const Tensor<float> doubled = r * 2.0f;

    EXPECT_FLOAT_EQ(doubled(0), 6.0f);
    EXPECT_FLOAT_EQ(a(0, 0), 3.0f);
    EXPECT_FALSE(doubled.SharesStorage(a));
}

TEST(TensorViewOps, ContiguityReportedCorrectly) {
    Tensor<float> a({2, 3}, 1.0f);
    EXPECT_TRUE(a.IsContiguous());
    EXPECT_TRUE(a.Row(0).IsContiguous());
    EXPECT_TRUE(a.Slice(0, 0, 1).IsContiguous());
    EXPECT_FALSE(a.Transpose().IsContiguous());
}

TEST(TensorViewOps, ViewOfAViewComposes) {
    Tensor<float> a({4, 4});
    IotaFill(a);

    const Tensor<float> block = a.Slice(0, 1, 2).Slice(1, 1, 2);

    EXPECT_EQ(block.Shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(block(0, 0), a(1, 1));
    EXPECT_FLOAT_EQ(block(1, 1), a(2, 2));
}

// --- contiguous -------------------------------------------------------------

TEST(TensorContiguous, AlreadyContiguousSharesStorage) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> c = a.Contiguous();

    EXPECT_TRUE(c.SharesStorage(a));
    EXPECT_EQ(c.Data(), a.Data());
    EXPECT_TRUE(c.IsContiguous());
}

TEST(TensorContiguous, NonContiguousIsCompactedIntoNewStorage) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> t = a.Transpose();
    ASSERT_FALSE(t.IsContiguous());

    const Tensor<float> c = t.Contiguous();

    EXPECT_TRUE(c.IsContiguous());
    EXPECT_FALSE(c.SharesStorage(a));
    EXPECT_EQ(c.Shape(), t.Shape());
    EXPECT_EQ(c.Strides(), (std::vector<size_t>{2, 1}));
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(c(i, j), t(i, j));
        }
    }
}

TEST(TensorContiguous, UnblocksViewOnATransposedTensor) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> t = a.Transpose();
    EXPECT_THROW(t.View(6), std::invalid_argument);

    const Tensor<float> flat = t.Contiguous().View(6);
    EXPECT_EQ(flat.Shape(), (std::vector<size_t>{6}));
    EXPECT_FLOAT_EQ(flat(0), a(0, 0));
    EXPECT_FLOAT_EQ(flat(1), a(1, 0));
}

TEST(TensorContiguous, CompactedCopyIsDecoupledFromTheSource) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<float> c = a.Transpose().Contiguous();

    c.Fill(9.0f);

    EXPECT_FLOAT_EQ(a(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(c(0, 0), 9.0f);
}

TEST(TensorContiguous, SliceOfRowsStaysContiguous) {
    Tensor<float> a({4, 3});
    IotaFill(a);

    const Tensor<float> s = a.Slice(0, 1, 2);
    ASSERT_TRUE(s.IsContiguous());

    const Tensor<float> c = s.Contiguous();
    EXPECT_TRUE(c.SharesStorage(a)) << "no copy needed when already contiguous";
}

TEST(TensorContiguous, EmptyTensorIsHandled) {
    Tensor<float> t;
    const Tensor<float> c = t.Contiguous();
    EXPECT_EQ(c.Size(), 0u);
}

// --- Reshape ----------------------------------------------------------------

TEST(TensorReshape, ContiguousSourceSharesStorage) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> r = a.Reshape(3, 2);

    EXPECT_TRUE(r.SharesStorage(a)) << "no copy needed for a contiguous source";
    EXPECT_EQ(r.Shape(), (std::vector<size_t>{3, 2}));
    EXPECT_EQ(r.Size(), a.Size());
    EXPECT_TRUE(r.IsContiguous());
}

TEST(TensorReshape, PreservesRowMajorOrder) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> flat = a.Reshape(6);

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(flat(i), static_cast<float>(i + 1)) << "at " << i;
    }
}

TEST(TensorReshape, WritesPropagateWhenSharing) {
    Tensor<float> a({2, 3}, 0.0f);
    Tensor<float> r = a.Reshape(6);

    r(5) = 42.0f;

    EXPECT_FLOAT_EQ(a(1, 2), 42.0f);
}

TEST(TensorReshape, ChangesRank) {
    Tensor<float> a({2, 3, 4});
    IotaFill(a);

    const Tensor<float> flat = a.Reshape(24);
    const Tensor<float> matrix = a.Reshape(6, 4);

    EXPECT_EQ(flat.Rank(), 1u);
    EXPECT_EQ(matrix.Shape(), (std::vector<size_t>{6, 4}));
    EXPECT_FLOAT_EQ(flat(23), matrix(5, 3));
}

TEST(TensorReshape, VectorAndVariadicFormsAgree) {
    Tensor<float> a({2, 6});
    IotaFill(a);

    const Tensor<float> v = a.Reshape(std::vector<size_t>{3, 4});
    const Tensor<float> d = a.Reshape(3, 4);

    EXPECT_EQ(v.Shape(), d.Shape());
    EXPECT_FLOAT_EQ(v(2, 3), d(2, 3));
}

TEST(TensorReshape, IdentityShapeIsAllowed) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> r = a.Reshape(2, 3);

    EXPECT_EQ(r.Shape(), a.Shape());
    EXPECT_TRUE(r.SharesStorage(a));
    EXPECT_FLOAT_EQ(r(1, 2), a(1, 2));
}

TEST(TensorReshape, RejectsElementCountChange) {
    Tensor<float> a({2, 3}, 1.0f);

    EXPECT_THROW(a.Reshape(2, 2), std::invalid_argument);
    EXPECT_THROW(a.Reshape(7), std::invalid_argument);
    EXPECT_THROW(a.Reshape(2, 3, 2), std::invalid_argument);
}

TEST(TensorReshape, RejectsZeroDimension) {
    Tensor<float> a({2, 3}, 1.0f);
    EXPECT_THROW(a.Reshape(0, 6), std::invalid_argument);
}

TEST(TensorReshape, RejectsEmptyTensor) {
    Tensor<float> empty;
    EXPECT_THROW(empty.Reshape(1), std::invalid_argument);
}

// --- Reshape over non-contiguous sources ------------------------------------

TEST(TensorReshape, NonContiguousSourceIsCopied) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> t = a.Transpose();
    ASSERT_FALSE(t.IsContiguous());

    const Tensor<float> r = t.Reshape(6);

    EXPECT_FALSE(r.SharesStorage(a)) << "a strided source must be compacted";
    EXPECT_TRUE(r.IsContiguous());
    EXPECT_EQ(r.Size(), 6u);
}

TEST(TensorReshape, NonContiguousCopyUsesLogicalOrder) {
    Tensor<float> a({2, 3});
    IotaFill(a);   // rows: (1 2 3), (4 5 6)

    const Tensor<float> t = a.Transpose();   // 3x2: (1 4), (2 5), (3 6)
    const Tensor<float> r = t.Reshape(6);

    const float expected[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(r(i), expected[i]) << "at " << i;
    }
}

TEST(TensorReshape, OffsetIsResetWhenCompacting) {
    // Regression: the clone path must drop the source offset, or every read
    // lands past the end of the freshly allocated buffer.
    Tensor<float> a({4, 4});
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            a(i, j) = static_cast<float>(i * 4 + j);
        }
    }

    const Tensor<float> block = a.Slice(0, 1, 2).Slice(1, 1, 2);
    ASSERT_FALSE(block.IsContiguous());
    ASSERT_GT(block.Offset(), 0u);

    const Tensor<float> r = block.Reshape(4);

    EXPECT_EQ(r.Offset(), 0u);
    EXPECT_FLOAT_EQ(r(0), 5.0f);
    EXPECT_FLOAT_EQ(r(1), 6.0f);
    EXPECT_FLOAT_EQ(r(2), 9.0f);
    EXPECT_FLOAT_EQ(r(3), 10.0f);
}

TEST(TensorReshape, ContiguousViewWithOffsetKeepsItsOffset) {
    Tensor<float> a({3, 4});
    IotaFill(a);

    const Tensor<float> r = a.Row(1);
    ASSERT_TRUE(r.IsContiguous());
    ASSERT_EQ(r.Offset(), 4u);

    const Tensor<float> reshaped = r.Reshape(2, 2);

    EXPECT_TRUE(reshaped.SharesStorage(a));
    EXPECT_EQ(reshaped.Offset(), 4u);
    EXPECT_FLOAT_EQ(reshaped(0, 0), a(1, 0));
    EXPECT_FLOAT_EQ(reshaped(1, 1), a(1, 3));
}

TEST(TensorReshape, CopiedResultIsDecoupledFromTheSource) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<float> r = a.Transpose().Reshape(6);

    r.Fill(9.0f);

    EXPECT_FLOAT_EQ(a(0, 0), 1.0f) << "compacted copy must not alias the source";
}

TEST(TensorReshape, SucceedsWhereViewWouldThrow) {
    Tensor<float> a({2, 3}, 1.0f);
    const Tensor<float> t = a.Transpose();

    EXPECT_THROW(t.View(6), std::invalid_argument);
    EXPECT_NO_THROW(t.Reshape(6)) << "Reshape falls back to a copy";
}

TEST(TensorReshape, ResultIsUsableInArithmetic) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> r = a.Transpose().Reshape(3, 2);
    const Tensor<float> sum = r + r;

    EXPECT_EQ(sum.Shape(), (std::vector<size_t>{3, 2}));
    EXPECT_FLOAT_EQ(sum(0, 0), r(0, 0) * 2.0f);
    EXPECT_FLOAT_EQ(sum(2, 1), r(2, 1) * 2.0f);
}

TEST(TensorReshape, ChainsWithOtherViews) {
    Tensor<float> a({4, 4});
    IotaFill(a);

    const Tensor<float> r = a.Slice(0, 1, 2).Reshape(8);

    EXPECT_EQ(r.Size(), 8u);
    EXPECT_TRUE(r.SharesStorage(a)) << "a row slice is still contiguous";
    EXPECT_FLOAT_EQ(r(0), a(1, 0));
    EXPECT_FLOAT_EQ(r(7), a(2, 3));
}

// --- Sqrt -------------------------------------------------------------------

TEST(TensorSqrt, AppliesElementwise) {
    Tensor<float> a({2, 2});
    a(0, 0) = 1.0f; a(0, 1) = 4.0f;
    a(1, 0) = 9.0f; a(1, 1) = 16.0f;

    const Tensor<float> r = a.Sqrt();

    EXPECT_FLOAT_EQ(r(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(r(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(r(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(r(1, 1), 4.0f);
}

TEST(TensorSqrt, DoesNotMutateOrAliasTheSource) {
    Tensor<float> a({1, 2}, 9.0f);

    const Tensor<float> r = a.Sqrt();

    EXPECT_FLOAT_EQ(a(0, 0), 9.0f);
    EXPECT_FALSE(r.SharesStorage(a));
}

TEST(TensorSqrt, HandlesANonContiguousSource) {
    Tensor<float> a({2, 2});
    a(0, 0) = 1.0f; a(0, 1) = 4.0f;
    a(1, 0) = 9.0f; a(1, 1) = 16.0f;

    const Tensor<float> t = a.Transpose();
    ASSERT_FALSE(t.IsContiguous());

    const Tensor<float> r = t.Sqrt();

    EXPECT_EQ(r.Shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(r(0, 1), 3.0f) << "t(0,1) is a(1,0) == 9";
    EXPECT_FLOAT_EQ(r(1, 0), 2.0f) << "t(1,0) is a(0,1) == 4";
}

TEST(TensorSqrt, FreeFormForwardsToTheMember) {
    Tensor<float> a({1, 2}, 25.0f);
    EXPECT_FLOAT_EQ(tensor::Sqrt(a)(0, 0), a.Sqrt()(0, 0));
}

// --- Mean over the whole tensor ---------------------------------------------

TEST(TensorMeanAll, AveragesEveryElement) {
    Tensor<float> a({2, 3});
    IotaFill(a);   // 1..6

    const Tensor<float> m = a.Mean();

    EXPECT_EQ(m.Shape(), (std::vector<size_t>{1}));
    EXPECT_FLOAT_EQ(m(0), 3.5f);
}

TEST(TensorMeanAll, ThrowsOnAnEmptyTensor) {
    Tensor<float> empty;
    EXPECT_THROW(empty.Mean(), std::invalid_argument);
}

TEST(TensorMeanAll, FreeFormForwardsToTheMember) {
    Tensor<float> a({2, 3});
    IotaFill(a);
    EXPECT_FLOAT_EQ(tensor::Mean(a)(0), a.Mean()(0));
}

// --- Mean over one axis -----------------------------------------------------

TEST(TensorMeanAxis, ReducingAxisZeroAveragesDownColumns) {
    Tensor<float> a({2, 3});
    IotaFill(a);   // rows (1 2 3), (4 5 6)

    const Tensor<float> m = a.Mean(0);

    EXPECT_EQ(m.Shape(), (std::vector<size_t>{3}));
    EXPECT_FLOAT_EQ(m(0), 2.5f);
    EXPECT_FLOAT_EQ(m(1), 3.5f);
    EXPECT_FLOAT_EQ(m(2), 4.5f);
}

TEST(TensorMeanAxis, ReducingAxisOneAveragesAcrossRows) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> m = a.Mean(1);

    EXPECT_EQ(m.Shape(), (std::vector<size_t>{2}));
    EXPECT_FLOAT_EQ(m(0), 2.0f);
    EXPECT_FLOAT_EQ(m(1), 5.0f);
}

TEST(TensorMeanAxis, KeepdimKeepsTheRank) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> m = a.Mean(1, true);

    EXPECT_EQ(m.Shape(), (std::vector<size_t>{2, 1}));
    EXPECT_EQ(m.Rank(), 2u);
    EXPECT_FLOAT_EQ(m(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(m(1, 0), 5.0f);
}

TEST(TensorMeanAxis, KeepdimResultBroadcastsAgainstTheSource) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> centred = a - a.Mean(1, true);

    ASSERT_EQ(centred.Shape(), (std::vector<size_t>{2, 3}));
    for (size_t i = 0; i < 2; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < 3; ++j) sum += centred(i, j);
        EXPECT_NEAR(sum, 0.0f, 1e-5f) << "row " << i << " must be centred";
    }
}

TEST(TensorMeanAxis, RankThreeReducesTheMiddleAxis) {
    Tensor<float> a({2, 3, 4});
    IotaFill(a);

    const Tensor<float> m = a.Mean(1);

    ASSERT_EQ(m.Shape(), (std::vector<size_t>{2, 4}));
    for (size_t i = 0; i < 2; ++i) {
        for (size_t k = 0; k < 4; ++k) {
            const float expected = (a(i, 0, k) + a(i, 1, k) + a(i, 2, k)) / 3.0f;
            EXPECT_FLOAT_EQ(m(i, k), expected) << "at (" << i << ", " << k << ")";
        }
    }
}

TEST(TensorMeanAxis, RankThreeReducesTheLastAxis) {
    Tensor<float> a({2, 3, 4});
    IotaFill(a);

    const Tensor<float> m = a.Mean(2, true);

    ASSERT_EQ(m.Shape(), (std::vector<size_t>{2, 3, 1}));
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < 4; ++k) sum += a(i, j, k);
            EXPECT_FLOAT_EQ(m(i, j, 0), sum / 4.0f);
        }
    }
}

TEST(TensorMeanAxis, RankOneCollapsesToAScalar) {
    Tensor<float> a({4});
    for (size_t i = 0; i < 4; ++i) a(i) = static_cast<float>(i + 1);

    const Tensor<float> m = a.Mean(0);

    EXPECT_EQ(m.Rank(), 0u);
    EXPECT_EQ(m.Size(), 1u);
    EXPECT_FLOAT_EQ(m(), 2.5f);
}

TEST(TensorMeanAxis, ReducingALengthOneAxisIsIdentity) {
    Tensor<float> a({1, 3});
    IotaFill(a);

    const Tensor<float> m = a.Mean(0);

    EXPECT_EQ(m.Shape(), (std::vector<size_t>{3}));
    EXPECT_FLOAT_EQ(m(0), 1.0f);
    EXPECT_FLOAT_EQ(m(2), 3.0f);
}

TEST(TensorMeanAxis, HandlesANonContiguousSource) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> t = a.Transpose();   // {3,2}
    ASSERT_FALSE(t.IsContiguous());

    const Tensor<float> m = t.Mean(0);       // average down each column of t

    EXPECT_EQ(m.Shape(), (std::vector<size_t>{2}));
    EXPECT_FLOAT_EQ(m(0), 2.0f) << "column 0 of t is row 0 of a";
    EXPECT_FLOAT_EQ(m(1), 5.0f);
}

TEST(TensorMeanAxis, DoesNotMutateOrAliasTheSource) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> m = a.Mean(1, true);

    EXPECT_FALSE(m.SharesStorage(a));
    EXPECT_FLOAT_EQ(a(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(a(1, 2), 6.0f);
}

TEST(TensorMeanAxis, RejectsAnOutOfRangeAxis) {
    Tensor<float> a({2, 3}, 1.0f);

    EXPECT_THROW(a.Mean(2), std::invalid_argument);
    EXPECT_THROW(a.Mean(static_cast<size_t>(-1)), std::invalid_argument);
}

TEST(TensorMeanAxis, FreeFormForwardsToTheMember) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    EXPECT_FLOAT_EQ(tensor::Mean(a, 1)(0), a.Mean(1)(0));
    EXPECT_EQ(tensor::Mean(a, 1, true).Shape(), a.Mean(1, true).Shape());
}

TEST(TensorMeanAxis, WorksWithDouble) {
    Tensor<double> a({2, 2});
    a(0, 0) = 1.0; a(0, 1) = 2.0;
    a(1, 0) = 3.0; a(1, 1) = 4.0;

    const Tensor<double> m = a.Mean(0);

    EXPECT_DOUBLE_EQ(m(0), 2.0);
    EXPECT_DOUBLE_EQ(m(1), 3.0);
}

// --- scalar add / subtract --------------------------------------------------

TEST(TensorScalarArithmetic, AddAssignShiftsEveryElement) {
    Tensor<float> a({2, 3}, 1.0f);
    a += 4.0f;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a(i, j), 5.0f);
        }
    }
}

TEST(TensorScalarArithmetic, SubtractAssignShiftsEveryElement) {
    Tensor<float> a({2, 2}, 10.0f);
    a -= 3.0f;

    EXPECT_FLOAT_EQ(a(0, 0), 7.0f);
    EXPECT_FLOAT_EQ(a(1, 1), 7.0f);
}

TEST(TensorScalarArithmetic, CompoundFormsReturnAReference) {
    Tensor<float> a({1, 2}, 0.0f);
    (a += 1.0f) += 2.0f;

    EXPECT_FLOAT_EQ(a(0, 0), 3.0f);
}

TEST(TensorScalarArithmetic, AddsScalarOnTheRight) {
    Tensor<float> a({2, 2}, 1.0f);

    const Tensor<float> r = a + 2.0f;

    EXPECT_FLOAT_EQ(r(0, 0), 3.0f);
    EXPECT_FLOAT_EQ(a(0, 0), 1.0f) << "operand must be untouched";
    EXPECT_FALSE(r.SharesStorage(a));
}

TEST(TensorScalarArithmetic, AddsScalarOnTheLeft) {
    Tensor<float> a({2, 2}, 1.0f);

    const Tensor<float> r = 2.0f + a;

    EXPECT_FLOAT_EQ(r(0, 0), 3.0f);
    EXPECT_FLOAT_EQ(a(0, 0), 1.0f);
}

TEST(TensorScalarArithmetic, AdditionCommutes) {
    Tensor<float> a({2, 2}, 4.0f);
    EXPECT_FLOAT_EQ((a + 3.0f)(1, 1), (3.0f + a)(1, 1));
}

TEST(TensorScalarArithmetic, SubtractsScalarOnTheRight) {
    Tensor<float> a({2, 2}, 10.0f);

    const Tensor<float> r = a - 4.0f;

    EXPECT_FLOAT_EQ(r(0, 0), 6.0f);
    EXPECT_FLOAT_EQ(a(0, 0), 10.0f);
}

TEST(TensorScalarArithmetic, SubtractionDoesNotCommute) {
    // 10 - t must be 10 - t[i], not t[i] - 10.
    Tensor<float> a({2, 2}, 4.0f);

    const Tensor<float> right = a - 10.0f;
    const Tensor<float> left = 10.0f - a;

    EXPECT_FLOAT_EQ(right(0, 0), -6.0f);
    EXPECT_FLOAT_EQ(left(0, 0), 6.0f);
}

TEST(TensorScalarArithmetic, ScalarSubtractHandlesMixedSigns) {
    Tensor<float> a({1, 3});
    a(0, 0) = -2.0f; a(0, 1) = 0.0f; a(0, 2) = 5.0f;

    const Tensor<float> r = 1.0f - a;

    EXPECT_FLOAT_EQ(r(0, 0), 3.0f);
    EXPECT_FLOAT_EQ(r(0, 1), 1.0f);
    EXPECT_FLOAT_EQ(r(0, 2), -4.0f);
}

TEST(TensorScalarArithmetic, WorksThroughANonContiguousView) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    Tensor<float> t = a.Transpose();
    t += 10.0f;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a(i, j), static_cast<float>(i * 3 + j + 1) + 10.0f)
                << "at (" << i << ", " << j << ")";
        }
    }
}

TEST(TensorScalarArithmetic, PreservesShape) {
    Tensor<float> a({2, 3, 4}, 1.0f);

    EXPECT_EQ((a + 1.0f).Shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_EQ((1.0f - a).Shape(), (std::vector<size_t>{2, 3, 4}));
}

TEST(TensorScalarArithmetic, WorksWithDouble) {
    Tensor<double> a({1, 2}, 1.5);

    EXPECT_DOUBLE_EQ((a + 0.5)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((0.5 - a)(0, 0), -1.0);
}

// --- batched matmul ---------------------------------------------------------

TEST(TensorMatMulBatched, RankThreeMultipliesEachBatchIndependently) {
    // batch 0: [[1 2],[3 4]]   batch 1: [[5 6],[7 8]]
    Tensor<float> a({2, 2, 2});
    IotaFill(a);
    Tensor<float> b({2, 2, 2});
    IotaFill(b);

    const Tensor<float> c = tensor::MatMul(a, b);

    ASSERT_EQ(c.Shape(), (std::vector<size_t>{2, 2, 2}));

    // [[1 2],[3 4]] x [[1 2],[3 4]] = [[7 10],[15 22]]
    EXPECT_FLOAT_EQ(c(0, 0, 0), 7.0f);
    EXPECT_FLOAT_EQ(c(0, 0, 1), 10.0f);
    EXPECT_FLOAT_EQ(c(0, 1, 0), 15.0f);
    EXPECT_FLOAT_EQ(c(0, 1, 1), 22.0f);

    // [[5 6],[7 8]] x [[5 6],[7 8]] = [[67 78],[91 106]]
    EXPECT_FLOAT_EQ(c(1, 0, 0), 67.0f);
    EXPECT_FLOAT_EQ(c(1, 0, 1), 78.0f);
    EXPECT_FLOAT_EQ(c(1, 1, 0), 91.0f);
    EXPECT_FLOAT_EQ(c(1, 1, 1), 106.0f);
}

TEST(TensorMatMulBatched, BatchesDoNotLeakIntoEachOther) {
    Tensor<float> a({2, 1, 2}, 0.0f);
    a(0, 0, 0) = 1.0f; a(0, 0, 1) = 1.0f;
    a(1, 0, 0) = 100.0f; a(1, 0, 1) = 100.0f;

    Tensor<float> b({2, 2, 1}, 1.0f);

    const Tensor<float> c = tensor::MatMul(a, b);

    ASSERT_EQ(c.Shape(), (std::vector<size_t>{2, 1, 1}));
    EXPECT_FLOAT_EQ(c(0, 0, 0), 2.0f);
    EXPECT_FLOAT_EQ(c(1, 0, 0), 200.0f);
}

TEST(TensorMatMulBatched, NonSquareShapesPropagate) {
    Tensor<float> a({3, 2, 4}, 1.0f);
    Tensor<float> b({3, 4, 5}, 1.0f);

    const Tensor<float> c = tensor::MatMul(a, b);

    EXPECT_EQ(c.Shape(), (std::vector<size_t>{3, 2, 5}));
    EXPECT_FLOAT_EQ(c(0, 0, 0), 4.0f) << "sum over the shared dimension of 4";
}

TEST(TensorMatMulBatched, RankFourBatchesOverTwoLeadingAxes) {
    Tensor<float> a({2, 3, 2, 2}, 1.0f);
    Tensor<float> b({2, 3, 2, 2}, 2.0f);

    const Tensor<float> c = tensor::MatMul(a, b);

    EXPECT_EQ(c.Shape(), (std::vector<size_t>{2, 3, 2, 2}));
    EXPECT_FLOAT_EQ(c(1, 2, 1, 1), 4.0f);
}

TEST(TensorMatMulBatched, BatchDimensionsBroadcast) {
    // A rank-2 operand applies to every batch of the rank-3 one.
    Tensor<float> a({3, 2, 2}, 1.0f);
    Tensor<float> b({2, 2}, 2.0f);

    const Tensor<float> c = tensor::MatMul(a, b);

    ASSERT_EQ(c.Shape(), (std::vector<size_t>{3, 2, 2}));
    for (size_t batch = 0; batch < 3; ++batch) {
        EXPECT_FLOAT_EQ(c(batch, 0, 0), 4.0f) << "batch " << batch;
    }
}

TEST(TensorMatMulBatched, BroadcastsALengthOneBatchAxis) {
    Tensor<float> a({1, 2, 2}, 1.0f);
    Tensor<float> b({4, 2, 2}, 1.0f);

    const Tensor<float> c = tensor::MatMul(a, b);

    EXPECT_EQ(c.Shape(), (std::vector<size_t>{4, 2, 2}));
    EXPECT_FLOAT_EQ(c(3, 1, 1), 2.0f);
}

TEST(TensorMatMulBatched, RejectsIncompatibleBatchShapes) {
    Tensor<float> a({2, 2, 2}, 1.0f);
    Tensor<float> b({3, 2, 2}, 1.0f);

    EXPECT_THROW(tensor::MatMul(a, b), std::invalid_argument);
}

TEST(TensorMatMulBatched, RejectsMismatchedInnerDimension) {
    Tensor<float> a({2, 2, 3}, 1.0f);
    Tensor<float> b({2, 4, 2}, 1.0f);

    EXPECT_THROW(tensor::MatMul(a, b), std::invalid_argument);
}

TEST(TensorMatMulBatched, MatchesPerBatchTwoDimensionalResults) {
    Tensor<float> a({2, 2, 3});
    IotaFill(a);
    Tensor<float> b({2, 3, 2});
    IotaFill(b);

    const Tensor<float> batched = tensor::MatMul(a, b);

    for (size_t batch = 0; batch < 2; ++batch) {
        Tensor<float> a2({2, 3});
        Tensor<float> b2({3, 2});
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j) a2(i, j) = a(batch, i, j);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 2; ++j) b2(i, j) = b(batch, i, j);

        const Tensor<float> expected = tensor::MatMul(a2, b2);
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                EXPECT_FLOAT_EQ(batched(batch, i, j), expected(i, j))
                    << "batch " << batch << " at (" << i << ", " << j << ")";
            }
        }
    }
}

TEST(TensorMatMulBatched, AcceptsATransposedViewAsTheRightOperand) {
    // The shape attention needs: {heads, seq, dim} x {heads, dim, seq}.
    Tensor<float> q({2, 3, 4});
    IotaFill(q);

    const Tensor<float> scores = tensor::MatMul(q, q.Transpose(1, 2));

    ASSERT_EQ(scores.Shape(), (std::vector<size_t>{2, 3, 3}));

    // Each entry is a dot product of two rows of the same head.
    for (size_t h = 0; h < 2; ++h) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                float expected = 0.0f;
                for (size_t d = 0; d < 4; ++d) expected += q(h, i, d) * q(h, j, d);
                EXPECT_FLOAT_EQ(scores(h, i, j), expected)
                    << "head " << h << " at (" << i << ", " << j << ")";
            }
        }
    }
}

TEST(TensorMatMulBatched, ScoreMatrixIsSymmetricForSelfAttention) {
    Tensor<float> q({1, 3, 2});
    IotaFill(q);

    const Tensor<float> scores = tensor::MatMul(q, q.Transpose(1, 2));

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(scores(0, i, j), scores(0, j, i));
        }
    }
}

TEST(TensorMatMulBatched, HandlesNonContiguousLeftOperand) {
    Tensor<float> a({2, 4, 3});
    IotaFill(a);

    const Tensor<float> view = a.Transpose(1, 2);   // {2,3,4}, non-contiguous
    ASSERT_FALSE(view.IsContiguous());

    Tensor<float> b({2, 4, 1}, 1.0f);
    const Tensor<float> c = tensor::MatMul(view, b);

    ASSERT_EQ(c.Shape(), (std::vector<size_t>{2, 3, 1}));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t i = 0; i < 3; ++i) {
            float expected = 0.0f;
            for (size_t k = 0; k < 4; ++k) expected += view(batch, i, k);
            EXPECT_FLOAT_EQ(c(batch, i, 0), expected);
        }
    }
}

TEST(TensorMatMulBatched, StillRejectsRankOneOperands) {
    Tensor<float> vec({3}, 1.0f);
    Tensor<float> mat({3, 3}, 1.0f);

    EXPECT_THROW(tensor::MatMul(vec, mat), std::invalid_argument);
    EXPECT_THROW(tensor::MatMul(mat, vec), std::invalid_argument);
}

TEST(TensorMatMulBatched, WorksWithDouble) {
    Tensor<double> a({2, 2, 2}, 0.5);
    Tensor<double> b({2, 2, 2}, 4.0);

    const Tensor<double> c = tensor::MatMul(a, b);

    EXPECT_EQ(c.Shape(), (std::vector<size_t>{2, 2, 2}));
    EXPECT_DOUBLE_EQ(c(1, 0, 0), 4.0);
}

// --- negative axis indexing -------------------------------------------------

TEST(TensorTransposeNegative, MinusOneIsTheLastAxis) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> viaNegative = a.Transpose(-2, -1);
    const Tensor<float> viaPositive = a.Transpose(0, 1);

    EXPECT_EQ(viaNegative.Shape(), viaPositive.Shape());
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(viaNegative(i, j), viaPositive(i, j));
        }
    }
}

TEST(TensorTransposeNegative, SwapsTheLastTwoAxesAtRankThree) {
    Tensor<float> a({2, 3, 4});
    IotaFill(a);

    const Tensor<float> t = a.Transpose(-2, -1);

    ASSERT_EQ(t.Shape(), (std::vector<size_t>{2, 4, 3}));
    for (size_t h = 0; h < 2; ++h)
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                EXPECT_FLOAT_EQ(t(h, j, i), a(h, i, j));
}

TEST(TensorTransposeNegative, SameExpressionWorksAtAnyRank) {
    // The point of -2,-1: identical code for {H,S,D} and {B,H,S,D}.
    Tensor<float> rank3({2, 3, 4}, 1.0f);
    Tensor<float> rank4({2, 3, 4, 5}, 1.0f);

    EXPECT_EQ(rank3.Transpose(-2, -1).Shape(), (std::vector<size_t>{2, 4, 3}));
    EXPECT_EQ(rank4.Transpose(-2, -1).Shape(), (std::vector<size_t>{2, 3, 5, 4}));
}

TEST(TensorTransposeNegative, MixesSignedAndUnsignedAxes) {
    Tensor<float> a({2, 3, 4});
    IotaFill(a);

    const Tensor<float> mixed = a.Transpose(0, -1);
    const Tensor<float> plain = a.Transpose(0, 2);

    EXPECT_EQ(mixed.Shape(), plain.Shape());
    EXPECT_FLOAT_EQ(mixed(3, 2, 1), plain(3, 2, 1));
}

TEST(TensorTransposeNegative, StillSharesStorage) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    EXPECT_TRUE(a.Transpose(-2, -1).SharesStorage(a));
}

TEST(TensorTransposeNegative, RejectsOutOfRangeNegativeAxes) {
    Tensor<float> a({2, 3}, 1.0f);

    EXPECT_THROW(a.Transpose(-3, 0), std::invalid_argument);
    EXPECT_THROW(a.Transpose(0, -4), std::invalid_argument);
}

TEST(TensorTransposeNegative, FreeFormAcceptsNegativeAxes) {
    Tensor<float> a({2, 3, 4});
    IotaFill(a);

    EXPECT_EQ(tensor::Transpose(a, -2, -1).Shape(), (std::vector<size_t>{2, 4, 3}));
}

TEST(TensorTransposeNegative, FeedsBatchedMatMulDirectly) {
    // The attention expression, rank-agnostic.
    Tensor<float> q({2, 3, 4});
    IotaFill(q);

    const Tensor<float> scores = tensor::MatMul(q, q.Transpose(-2, -1));

    ASSERT_EQ(scores.Shape(), (std::vector<size_t>{2, 3, 3}));
    for (size_t h = 0; h < 2; ++h) {
        for (size_t i = 0; i < 3; ++i) {
            float expected = 0.0f;
            for (size_t d = 0; d < 4; ++d) expected += q(h, i, d) * q(h, i, d);
            EXPECT_FLOAT_EQ(scores(h, i, i), expected);
        }
    }
}

// --- MaskedFill -------------------------------------------------------------

TEST(TensorMaskedFill, ReplacesOnlyMaskedPositions) {
    Tensor<float> a({2, 3}, 5.0f);
    Tensor<bool> mask({2, 3}, false);
    mask(0, 1) = true;
    mask(1, 2) = true;

    const Tensor<float> r = a.MaskedFill(mask, -1.0f);

    EXPECT_FLOAT_EQ(r(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(r(0, 1), -1.0f);
    EXPECT_FLOAT_EQ(r(0, 2), 5.0f);
    EXPECT_FLOAT_EQ(r(1, 0), 5.0f);
    EXPECT_FLOAT_EQ(r(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(r(1, 2), -1.0f);
}

TEST(TensorMaskedFill, AnAllFalseMaskChangesNothing) {
    Tensor<float> a({2, 3});
    IotaFill(a);
    Tensor<bool> mask({2, 3}, false);

    const Tensor<float> r = a.MaskedFill(mask, -99.0f);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(r(i, j), a(i, j));
        }
    }
}

TEST(TensorMaskedFill, AnAllTrueMaskReplacesEverything) {
    Tensor<float> a({2, 3});
    IotaFill(a);
    Tensor<bool> mask({2, 3}, true);

    const Tensor<float> r = a.MaskedFill(mask, 7.0f);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(r(i, j), 7.0f);
        }
    }
}

TEST(TensorMaskedFill, PreservesShape) {
    Tensor<float> a({2, 3, 4}, 1.0f);
    Tensor<bool> mask({2, 3, 4}, false);

    EXPECT_EQ(a.MaskedFill(mask, 0.0f).Shape(), (std::vector<size_t>{2, 3, 4}));
}

TEST(TensorMaskedFill, DoesNotMutateOrAliasTheSource) {
    Tensor<float> a({2, 3}, 5.0f);
    Tensor<bool> mask({2, 3}, true);

    const Tensor<float> r = a.MaskedFill(mask, -1.0f);

    EXPECT_FLOAT_EQ(a(0, 0), 5.0f) << "source must be untouched";
    EXPECT_FALSE(r.SharesStorage(a));
}

TEST(TensorMaskedFill, RejectsAMismatchedMaskShape) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<bool> wrong({3, 2}, false);
    Tensor<bool> smaller({2}, false);

    EXPECT_THROW(a.MaskedFill(wrong, 0.0f), std::invalid_argument);
    EXPECT_THROW(a.MaskedFill(smaller, 0.0f), std::invalid_argument);
}

TEST(TensorMaskedFill, AcceptsNegativeInfinity) {
    Tensor<float> a({1, 3}, 1.0f);
    Tensor<bool> mask({1, 3}, false);
    mask(0, 2) = true;

    const Tensor<float> r = a.MaskedFill(mask, -std::numeric_limits<float>::infinity());

    EXPECT_FLOAT_EQ(r(0, 0), 1.0f);
    EXPECT_TRUE(std::isinf(r(0, 2)));
    EXPECT_LT(r(0, 2), 0.0f);
}

TEST(TensorMaskedFill, BuildsACausalMask) {
    // The attention use: block every position from attending to later ones.
    const size_t seq = 4;
    Tensor<float> scores({seq, seq}, 1.0f);
    Tensor<bool> mask({seq, seq}, false);
    for (size_t i = 0; i < seq; ++i) {
        for (size_t j = i + 1; j < seq; ++j) {
            mask(i, j) = true;
        }
    }

    const Tensor<float> r = scores.MaskedFill(mask, -std::numeric_limits<float>::infinity());

    for (size_t i = 0; i < seq; ++i) {
        for (size_t j = 0; j < seq; ++j) {
            if (j <= i) {
                EXPECT_FLOAT_EQ(r(i, j), 1.0f) << "(" << i << "," << j << ") should be visible";
            } else {
                EXPECT_TRUE(std::isinf(r(i, j))) << "(" << i << "," << j << ") should be masked";
            }
        }
    }
}

TEST(TensorMaskedFill, HandlesANonContiguousSource) {
    Tensor<float> a({2, 3});
    IotaFill(a);                      // rows (1 2 3), (4 5 6)

    const Tensor<float> view = a.Transpose();   // {3,2}: (1 4), (2 5), (3 6)
    ASSERT_FALSE(view.IsContiguous());

    Tensor<bool> mask({3, 2}, false);
    mask(0, 0) = true;                // should hit the value 1
    mask(2, 1) = true;                // should hit the value 6

    const Tensor<float> r = view.MaskedFill(mask, -1.0f);

    EXPECT_FLOAT_EQ(r(0, 0), -1.0f);
    EXPECT_FLOAT_EQ(r(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(r(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(r(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(r(2, 0), 3.0f);
    EXPECT_FLOAT_EQ(r(2, 1), -1.0f);
}

TEST(TensorMaskedFill, HandlesANonContiguousMask) {
    Tensor<float> a({2, 3}, 5.0f);

    Tensor<bool> source({3, 2}, false);
    source(1, 0) = true;              // transposes to (0,1)
    const Tensor<bool> mask = source.Transpose();   // {2,3}
    ASSERT_FALSE(mask.IsContiguous());

    const Tensor<float> r = a.MaskedFill(mask, -1.0f);

    EXPECT_FLOAT_EQ(r(0, 1), -1.0f) << "the one true position";
    EXPECT_FLOAT_EQ(r(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(r(1, 2), 5.0f);
}

TEST(TensorMaskedFill, WorksWithDouble) {
    Tensor<double> a({1, 2}, 2.5);
    Tensor<bool> mask({1, 2}, false);
    mask(0, 1) = true;

    const Tensor<double> r = a.MaskedFill(mask, 0.0);

    EXPECT_DOUBLE_EQ(r(0, 0), 2.5);
    EXPECT_DOUBLE_EQ(r(0, 1), 0.0);
}

// --- MaskedFill broadcasting ------------------------------------------------

TEST(TensorMaskedFillBroadcast, LowerRankMaskAppliesToEveryBatch) {
    Tensor<float> a({2, 3, 4}, 1.0f);
    Tensor<bool> mask({4}, false);
    mask(1) = true;

    const Tensor<float> r = a.MaskedFill(mask, -1.0f);

    ASSERT_EQ(r.Shape(), (std::vector<size_t>{2, 3, 4}));
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(r(i, j, 0), 1.0f);
            EXPECT_FLOAT_EQ(r(i, j, 1), -1.0f) << "at (" << i << "," << j << ")";
            EXPECT_FLOAT_EQ(r(i, j, 3), 1.0f);
        }
    }
}

TEST(TensorMaskedFillBroadcast, CausalMaskAppliesAcrossBatchAndHeads) {
    // The attention case: {S,S} mask over {B,H,S,S} scores.
    const size_t B = 2, H = 3, S = 4;
    Tensor<float> scores({B, H, S, S}, 1.0f);

    Tensor<bool> causal({S, S}, false);
    for (size_t i = 0; i < S; ++i) {
        for (size_t j = i + 1; j < S; ++j) {
            causal(i, j) = true;
        }
    }

    const Tensor<float> r = scores.MaskedFill(causal, -1.0f);

    ASSERT_EQ(r.Shape(), (std::vector<size_t>{B, H, S, S}));
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < S; ++i) {
                for (size_t j = 0; j < S; ++j) {
                    const float expected = (j > i) ? -1.0f : 1.0f;
                    EXPECT_FLOAT_EQ(r(b, h, i, j), expected)
                        << "b" << b << " h" << h << " (" << i << "," << j << ")";
                }
            }
        }
    }
}

TEST(TensorMaskedFillBroadcast, LengthOneAxisBroadcasts) {
    Tensor<float> a({3, 2}, 5.0f);
    Tensor<bool> mask({1, 2}, false);
    mask(0, 1) = true;

    const Tensor<float> r = a.MaskedFill(mask, 0.0f);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(r(i, 0), 5.0f);
        EXPECT_FLOAT_EQ(r(i, 1), 0.0f) << "row " << i;
    }
}

TEST(TensorMaskedFillBroadcast, ColumnMaskBroadcastsAlongRows) {
    Tensor<float> a({2, 3}, 5.0f);
    Tensor<bool> mask({2, 1}, false);
    mask(1, 0) = true;

    const Tensor<float> r = a.MaskedFill(mask, 0.0f);

    for (size_t j = 0; j < 3; ++j) {
        EXPECT_FLOAT_EQ(r(0, j), 5.0f);
        EXPECT_FLOAT_EQ(r(1, j), 0.0f);
    }
}

TEST(TensorMaskedFillBroadcast, RejectsAMaskThatWouldGrowTheTensor) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<bool> bigger({4, 2, 3}, false);

    EXPECT_THROW(a.MaskedFill(bigger, 0.0f), std::invalid_argument);
}

TEST(TensorMaskedFillBroadcast, RejectsAnIncompatibleMask) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<bool> wrong({2, 4}, false);

    EXPECT_THROW(a.MaskedFill(wrong, 0.0f), std::invalid_argument);
}

TEST(TensorMaskedFillBroadcast, RejectsAnEmptyMask) {
    Tensor<float> a({2, 3}, 1.0f);
    Tensor<bool> empty;

    EXPECT_THROW(a.MaskedFill(empty, 0.0f), std::invalid_argument);
}

TEST(TensorMaskedFillBroadcast, BroadcastMaskMayBeNonContiguous) {
    // {3,2} transposed gives a genuinely strided {2,3} mask. A {3,1} source
    // would not: a length-1 axis leaves the layout contiguous either way.
    Tensor<bool> source({3, 2}, false);
    source(1, 1) = true;
    const Tensor<bool> mask = source.Transpose();   // {2,3}, mask(1,1) true
    ASSERT_FALSE(mask.IsContiguous());

    Tensor<float> a({4, 2, 3}, 5.0f);
    const Tensor<float> r = a.MaskedFill(mask, 0.0f);

    ASSERT_EQ(r.Shape(), (std::vector<size_t>{4, 2, 3}));
    for (size_t b = 0; b < 4; ++b) {
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                const float expected = (i == 1 && j == 1) ? 0.0f : 5.0f;
                EXPECT_FLOAT_EQ(r(b, i, j), expected)
                    << "b" << b << " (" << i << "," << j << ")";
            }
        }
    }
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

// --- CopyInto ---------------------------------------------------------------

namespace {

// Rank-agnostic 1, 2, 3, ... in row-major order (IotaFill above stops at rank 3).
template <typename T>
void Iota(Tensor<T>& t) {
    std::vector<size_t> index(t.Rank(), 0);
    T next = T(1);

    for (size_t i = 0; i < t.Size(); ++i) {
        t(std::span<const size_t>(index)) = next++;
        for (size_t axis = t.Rank(); axis-- > 0; ) {
            if (++index[axis] < t.Shape()[axis]) break;
            index[axis] = 0;
        }
    }
}

}  // namespace

TEST(TensorCopyInto, FillsAContiguousRow) {
    Tensor<float> dst({4, 3});
    Tensor<float> src({1, 3});
    src(0, 0) = 7.0f; src(0, 1) = 8.0f; src(0, 2) = 9.0f;

    tensor::CopyInto(dst.Slice(0, 2, 1), src);

    const float expected[4][3] = {{0, 0, 0}, {0, 0, 0}, {7, 8, 9}, {0, 0, 0}};
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(dst(i, j), expected[i][j]) << "at (" << i << ", " << j << ")";
}

TEST(TensorCopyInto, WritesThroughToTheParentStorage) {
    Tensor<float> dst({2, 2});
    Tensor<float> view = dst.Slice(0, 1, 1);
    Tensor<float> src({1, 2}, 5.0f);

    tensor::CopyInto(view, src);

    EXPECT_FLOAT_EQ(dst(1, 0), 5.0f) << "CopyInto must not copy the view";
    EXPECT_FLOAT_EQ(dst(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(dst(0, 0), 0.0f);
}

TEST(TensorCopyInto, SourceIsUnchanged) {
    Tensor<float> dst({2, 2}, 1.0f);
    Tensor<float> src({2, 2});
    IotaFill(src);

    tensor::CopyInto(dst, src);

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_FLOAT_EQ(src(i, j), static_cast<float>(i * 2 + j + 1));
}

TEST(TensorCopyInto, DoesNotAliasSourceAndDestination) {
    Tensor<float> dst({2, 2});
    Tensor<float> src({2, 2}, 3.0f);

    tensor::CopyInto(dst, src);
    src.Fill(9.0f);

    EXPECT_FLOAT_EQ(dst(0, 0), 3.0f) << "CopyInto must copy values, not share storage";
}

// --- CopyInto: strided destinations -----------------------------------------

// The KV-cache append: one head-layout window inside a larger buffer. The
// destination is several disjoint runs, so a flat memcpy would be wrong.
TEST(TensorCopyInto, FillsOnePositionAcrossEveryHead) {
    constexpr size_t kHeads = 2, kMaxSeq = 5, kDim = 3;
    Tensor<float> cache({1, kHeads, kMaxSeq, kDim});
    Tensor<float> src({1, kHeads, 1, kDim});
    Iota(src);  // 1..6 in row-major order

    tensor::CopyInto(cache.Slice(2, 2, 1), src);

    for (size_t h = 0; h < kHeads; ++h)
        for (size_t s = 0; s < kMaxSeq; ++s)
            for (size_t d = 0; d < kDim; ++d) {
                const float expected = (s == 2) ? static_cast<float>(h * kDim + d + 1) : 0.0f;
                EXPECT_FLOAT_EQ(cache(0, h, s, d), expected)
                    << "h" << h << " s" << s << " d" << d;
            }
}

// With batch > 1 the head axis wraps *mid-copy*, so the carry has to rewind
// each side by its own stride -- and dst's head stride (max_seq * dim) differs
// from src's (seq * dim). This is the case a single-batch test cannot see.
TEST(TensorCopyInto, FillsOnePositionAcrossEveryHeadAndBatch) {
    constexpr size_t kBatch = 2, kHeads = 2, kMaxSeq = 5, kDim = 3;
    Tensor<float> cache({kBatch, kHeads, kMaxSeq, kDim}, -1.0f);
    Tensor<float> src({kBatch, kHeads, 1, kDim});
    Iota(src);  // 1..12

    tensor::CopyInto(cache.Slice(2, 2, 1), src);

    for (size_t b = 0; b < kBatch; ++b)
        for (size_t h = 0; h < kHeads; ++h)
            for (size_t s = 0; s < kMaxSeq; ++s)
                for (size_t d = 0; d < kDim; ++d) {
                    const float expected = (s == 2)
                        ? static_cast<float>(b * kHeads * kDim + h * kDim + d + 1)
                        : -1.0f;
                    EXPECT_FLOAT_EQ(cache(b, h, s, d), expected)
                        << "b" << b << " h" << h << " s" << s << " d" << d;
                }
}

TEST(TensorCopyInto, LeavesNeighbouringPositionsUntouched) {
    Tensor<float> cache({1, 2, 5, 3}, -1.0f);
    Tensor<float> src({1, 2, 1, 3}, 42.0f);

    tensor::CopyInto(cache.Slice(2, 2, 1), src);

    for (size_t h = 0; h < 2; ++h)
        for (size_t s = 0; s < 5; ++s)
            for (size_t d = 0; d < 3; ++d) {
                if (s == 2) continue;
                EXPECT_FLOAT_EQ(cache(0, h, s, d), -1.0f)
                    << "clobbered h" << h << " s" << s << " d" << d;
            }
}

TEST(TensorCopyInto, FillsAMultiPositionWindow) {
    Tensor<float> cache({1, 2, 5, 2}, -1.0f);
    Tensor<float> src({1, 2, 3, 2});
    Iota(src);  // 1..12

    tensor::CopyInto(cache.Slice(2, 1, 3), src);  // positions 1, 2, 3

    for (size_t h = 0; h < 2; ++h)
        for (size_t s = 0; s < 5; ++s)
            for (size_t d = 0; d < 2; ++d) {
                const bool inside = s >= 1 && s <= 3;
                const float expected = inside
                    ? static_cast<float>(h * 6 + (s - 1) * 2 + d + 1)
                    : -1.0f;
                EXPECT_FLOAT_EQ(cache(0, h, s, d), expected)
                    << "h" << h << " s" << s << " d" << d;
            }
}

TEST(TensorCopyInto, FillsAColumnWhoseElementsAreNotAdjacent) {
    Tensor<float> dst({4, 3}, -1.0f);
    Tensor<float> src({4, 1});
    IotaFill(src);  // 1..4

    tensor::CopyInto(dst.Slice(1, 1, 1), src);  // middle column only

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(dst(i, 0), -1.0f) << "row " << i;
        EXPECT_FLOAT_EQ(dst(i, 1), static_cast<float>(i + 1)) << "row " << i;
        EXPECT_FLOAT_EQ(dst(i, 2), -1.0f) << "row " << i;
    }
}

TEST(TensorCopyInto, AcceptsASlicedSourceAsWell) {
    Tensor<float> big({4, 3});
    IotaFill(big);  // 1..12
    Tensor<float> dst({2, 3}, -1.0f);

    tensor::CopyInto(dst, big.Slice(0, 1, 2));  // rows 1 and 2 -> 4..9

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(dst(i, j), static_cast<float>(4 + i * 3 + j))
                << "at (" << i << ", " << j << ")";
}

TEST(TensorCopyInto, HandlesBothSidesStridedAtOnce) {
    Tensor<float> src_parent({1, 2, 5, 3}, -1.0f);
    Tensor<float> dst_parent({1, 2, 5, 3}, -1.0f);

    Tensor<float> src = src_parent.Slice(2, 4, 1);  // last position
    Tensor<float> dst = dst_parent.Slice(2, 0, 1);  // first position
    for (size_t h = 0; h < 2; ++h)
        for (size_t d = 0; d < 3; ++d) src(0, h, 0, d) = static_cast<float>(h * 3 + d + 1);

    ASSERT_FALSE(src.IsContiguous());
    ASSERT_FALSE(dst.IsContiguous());
    tensor::CopyInto(dst, src);

    for (size_t h = 0; h < 2; ++h)
        for (size_t d = 0; d < 3; ++d)
            EXPECT_FLOAT_EQ(dst_parent(0, h, 0, d), static_cast<float>(h * 3 + d + 1));
}

TEST(TensorCopyInto, CopiesIntoATransposedView) {
    Tensor<float> dst({2, 3}, -1.0f);
    Tensor<float> src({3, 2});
    IotaFill(src);  // 1..6

    tensor::CopyInto(dst.Transpose(), src);  // dst^T is [3, 2] with strides [1, 3]

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_FLOAT_EQ(dst(j, i), src(i, j)) << "at (" << i << ", " << j << ")";
}

// --- CopyInto: edge cases ---------------------------------------------------

TEST(TensorCopyInto, RejectsAShapeMismatch) {
    Tensor<float> dst({2, 3});
    EXPECT_THROW(tensor::CopyInto(dst, Tensor<float>({2, 2})), std::invalid_argument);
    EXPECT_THROW(tensor::CopyInto(dst, Tensor<float>({3, 2})), std::invalid_argument);
    EXPECT_THROW(tensor::CopyInto(dst, Tensor<float>({6})), std::invalid_argument)
        << "same element count is not the same shape";
    EXPECT_THROW(tensor::CopyInto(dst, Tensor<float>({1, 2, 3})), std::invalid_argument)
        << "rank must match too";
}

TEST(TensorCopyInto, EmptyTensorsAreANoOp) {
    EXPECT_NO_THROW(tensor::CopyInto(Tensor<float>(), Tensor<float>()));
}

TEST(TensorCopyInto, CopiesASingleElement) {
    Tensor<float> dst({1, 1, 1, 1});
    tensor::CopyInto(dst, Tensor<float>({1, 1, 1, 1}, 4.0f));
    EXPECT_FLOAT_EQ(dst(0, 0, 0, 0), 4.0f);
}

TEST(TensorCopyInto, WholeTensorRoundTripsEveryElement) {
    Tensor<float> src({2, 3, 4});
    IotaFill(src);
    Tensor<float> dst({2, 3, 4}, -1.0f);

    tensor::CopyInto(dst, src);

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(dst(i, j, k), src(i, j, k))
                    << "at (" << i << ", " << j << ", " << k << ")";
}

TEST(TensorCopyInto, WorksForOtherElementTypes) {
    Tensor<int> dst({2, 2}, -1);
    tensor::CopyInto(dst.Slice(0, 1, 1), Tensor<int>({1, 2}, 6));
    EXPECT_EQ(dst(0, 0), -1);
    EXPECT_EQ(dst(1, 1), 6);
}

TEST(TensorUnsqueeze, InsertsSizeOneAxisAtEveryPosition) {
    Tensor<float> a({2, 3});

    EXPECT_EQ(a.Unsqueeze(0).Shape(), (std::vector<size_t>{1, 2, 3}));
    EXPECT_EQ(a.Unsqueeze(1).Shape(), (std::vector<size_t>{2, 1, 3}));
    EXPECT_EQ(a.Unsqueeze(2).Shape(), (std::vector<size_t>{2, 3, 1}));
}

TEST(TensorUnsqueeze, PreservesValuesAndElementCount) {
    Tensor<float> a({2, 3});
    IotaFill(a);

    const Tensor<float> u = a.Unsqueeze(1);

    EXPECT_EQ(u.Size(), a.Size());
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j) EXPECT_FLOAT_EQ(u(i, 0, j), a(i, j));
}

TEST(TensorUnsqueeze, IsAViewThatSharesStorage) {
    Tensor<float> a({2, 3}, 0.0f);
    Tensor<float> u = a.Unsqueeze(0);

    u(0, 1, 2) = 7.0f;

    EXPECT_TRUE(u.SharesStorage(a));
    EXPECT_FLOAT_EQ(a(1, 2), 7.0f);
}

TEST(TensorUnsqueeze, KeepsContiguousInputContiguous) {
    Tensor<float> a({2, 3});

    EXPECT_TRUE(a.Unsqueeze(0).IsContiguous());
    EXPECT_TRUE(a.Unsqueeze(1).IsContiguous());
    EXPECT_TRUE(a.Unsqueeze(2).IsContiguous());
}

TEST(TensorUnsqueeze, WorksOnNonContiguousSliceWithoutCopying) {
    Tensor<float> cache({2, 4, 3});
    IotaFill(cache);
    const Tensor<float> slice = cache.Slice(1, 0, 2);
    ASSERT_FALSE(slice.IsContiguous());

    Tensor<float> u = slice.Unsqueeze(1);

    EXPECT_TRUE(u.SharesStorage(cache));
    EXPECT_EQ(u.Shape(), (std::vector<size_t>{2, 1, 2, 3}));
    for (size_t h = 0; h < 2; ++h)
        for (size_t s = 0; s < 2; ++s)
            for (size_t d = 0; d < 3; ++d) EXPECT_FLOAT_EQ(u(h, 0, s, d), cache(h, s, d));

    u(1, 0, 1, 2) = -1.0f;
    EXPECT_FLOAT_EQ(cache(1, 1, 2), -1.0f);
}

TEST(TensorUnsqueeze, BroadcastsInMatMulWithoutCopying) {
    Tensor<float> kv({2, 4, 3});
    IotaFill(kv);
    const Tensor<float> k = kv.Slice(1, 0, 2).Unsqueeze(1);
    Tensor<float> q({2, 3, 1, 3}, 1.0f);

    const Tensor<float> scores = tensor::MatMul(q, k.Transpose(-2, -1));

    ASSERT_EQ(scores.Shape(), (std::vector<size_t>{2, 3, 1, 2}));
    for (size_t h = 0; h < 2; ++h)
        for (size_t g = 0; g < 3; ++g)
            for (size_t s = 0; s < 2; ++s)
                EXPECT_FLOAT_EQ(scores(h, g, 0, s), kv(h, s, 0) + kv(h, s, 1) + kv(h, s, 2));
}

TEST(TensorUnsqueeze, RejectsAxisPastTheEnd) {
    Tensor<float> a({2, 3});
    EXPECT_THROW(a.Unsqueeze(3), std::invalid_argument);
}

TEST(TensorUnsqueeze, RejectsEmptyTensor) {
    EXPECT_THROW(Tensor<float>().Unsqueeze(0), std::invalid_argument);
}

TEST(TensorSqueeze, RemovesSizeOneAxisAtEveryPosition) {
    EXPECT_EQ(Tensor<float>({1, 2, 3}).Squeeze(0).Shape(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(Tensor<float>({2, 1, 3}).Squeeze(1).Shape(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(Tensor<float>({2, 3, 1}).Squeeze(2).Shape(), (std::vector<size_t>{2, 3}));
}

TEST(TensorSqueeze, PreservesValuesAndSharesStorage) {
    Tensor<float> a({2, 1, 3});
    IotaFill(a);

    Tensor<float> s = a.Squeeze(1);

    EXPECT_TRUE(s.SharesStorage(a));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j) EXPECT_FLOAT_EQ(s(i, j), a(i, 0, j));

    s(1, 2) = -5.0f;
    EXPECT_FLOAT_EQ(a(1, 0, 2), -5.0f);
}

TEST(TensorSqueeze, WorksOnNonContiguousView) {
    Tensor<float> a({2, 4, 3});
    IotaFill(a);
    const Tensor<float> view = a.Slice(1, 1, 1);
    ASSERT_EQ(view.Shape(), (std::vector<size_t>{2, 1, 3}));

    const Tensor<float> s = view.Squeeze(1);

    EXPECT_TRUE(s.SharesStorage(a));
    EXPECT_EQ(s.Shape(), (std::vector<size_t>{2, 3}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j) EXPECT_FLOAT_EQ(s(i, j), a(i, 1, j));
}

TEST(TensorSqueeze, UndoesUnsqueeze) {
    Tensor<float> a({2, 4, 3});
    const Tensor<float> slice = a.Slice(1, 1, 2);

    for (size_t axis = 0; axis <= slice.Rank(); ++axis) {
        SCOPED_TRACE(axis);
        const Tensor<float> round_trip = slice.Unsqueeze(axis).Squeeze(axis);
        EXPECT_EQ(round_trip.Shape(), slice.Shape());
        EXPECT_EQ(round_trip.Strides(), slice.Strides());
        EXPECT_EQ(round_trip.Offset(), slice.Offset());
    }
}

TEST(TensorSqueeze, RejectsAxisLargerThanOne) {
    Tensor<float> a({2, 1, 3});
    EXPECT_THROW(a.Squeeze(0), std::invalid_argument);
    EXPECT_THROW(a.Squeeze(2), std::invalid_argument);
}

TEST(TensorSqueeze, RejectsOutOfRangeAxis) {
    Tensor<float> a({2, 1, 3});
    EXPECT_THROW(a.Squeeze(3), std::invalid_argument);
    EXPECT_THROW(a.Squeeze(4), std::invalid_argument);
}

TEST(TensorSqueeze, RejectsRemovingTheOnlyAxis) {
    Tensor<float> a({1});
    EXPECT_THROW(a.Squeeze(0), std::invalid_argument);
}

TEST(TensorSqueeze, RejectsEmptyTensor) {
    EXPECT_THROW(Tensor<float>().Squeeze(0), std::invalid_argument);
}

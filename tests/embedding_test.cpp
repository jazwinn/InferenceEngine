#include <gtest/gtest.h>

#include <NeuralNet/Embedding.h>
#include <NeuralNet/Module.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

using math::Tensor;
using nn::Embedding;
using nn::Module;

namespace {

// Row r of a [rows, dim] table holds r*10 + 0, r*10 + 1, ... so every
// looked-up value says which row and column it came from.
Tensor<float> NumberedTable(size_t rows, size_t dim) {
    Tensor<float> t({rows, dim});
    for (size_t r = 0; r < rows; ++r)
        for (size_t d = 0; d < dim; ++d) t(r, d) = static_cast<float>(r * 10 + d);
    return t;
}

Tensor<int64_t> Ids(std::initializer_list<int64_t> values) {
    Tensor<int64_t> t({values.size()});
    size_t i = 0;
    for (int64_t v : values) t(i++) = v;
    return t;
}

}  // namespace

// --- construction -----------------------------------------------------------

TEST(EmbeddingConstruction, WeightIsNumEmbeddingsByDim) {
    Embedding<float> emb(5, 3);

    ASSERT_TRUE(emb.weight);
    EXPECT_EQ(emb.weight->Shape(), (std::vector<size_t>{5, 3}));
    EXPECT_EQ(emb.NumEmbeddings(), 5u);
    EXPECT_EQ(emb.EmbeddingDim(), 3u);
}

TEST(EmbeddingConstruction, RegistersWeightAsAParameter) {
    Embedding<float> emb(5, 3);
    const auto named = emb.NamedParameters();

    ASSERT_EQ(named.size(), 1u);
    EXPECT_EQ(named[0].name, "weight");
    EXPECT_EQ(named[0].tensor, emb.weight);
}

TEST(EmbeddingConstruction, HasNoPaddingIndexByDefault) {
    EXPECT_FALSE(Embedding<float>(5, 3).PaddingIdx().has_value());
}

TEST(EmbeddingConstruction, AcceptsAPaddingIndex) {
    EXPECT_EQ(Embedding<float>(5, 3, 2).PaddingIdx(), 2u);
}

TEST(EmbeddingConstruction, ANegativePaddingIndexCountsFromTheEnd) {
    EXPECT_EQ(Embedding<float>(5, 3, -1).PaddingIdx(), 4u);
    EXPECT_EQ(Embedding<float>(5, 3, -5).PaddingIdx(), 0u);
}

TEST(EmbeddingConstruction, RejectsAPaddingIndexOutsideTheTable) {
    EXPECT_THROW(Embedding<float>(5, 3, 5), std::invalid_argument);
    EXPECT_THROW(Embedding<float>(5, 3, -6), std::invalid_argument);
}

TEST(EmbeddingConstruction, ThePaddingRowStartsAtZero) {
    Embedding<float> emb(5, 3, 1);
    for (size_t d = 0; d < 3; ++d) EXPECT_FLOAT_EQ((*emb.weight)(1, d), 0.0f);
}

TEST(EmbeddingConstruction, DefaultsToFloat) {
    static_assert(std::is_same_v<decltype(Embedding(4, 2))::value_type, float>);
    SUCCEED();
}

// --- FromPretrained ---------------------------------------------------------

TEST(EmbeddingFromPretrained, UsesTheGivenTable) {
    const Embedding<float> emb = Embedding<float>::FromPretrained(NumberedTable(4, 3));

    EXPECT_EQ(emb.NumEmbeddings(), 4u);
    EXPECT_EQ(emb.EmbeddingDim(), 3u);
    EXPECT_FLOAT_EQ((*emb.weight)(2, 1), 21.0f);
}

TEST(EmbeddingFromPretrained, TheRegisteredParameterIsTheLoadedTable) {
    const Embedding<float> emb = Embedding<float>::FromPretrained(NumberedTable(4, 3));
    const auto named = emb.NamedParameters();

    ASSERT_EQ(named.size(), 1u);
    EXPECT_EQ(named[0].tensor, emb.weight);
    EXPECT_FLOAT_EQ((*named[0].tensor)(3, 2), 32.0f);
}

TEST(EmbeddingFromPretrained, KeepsThePaddingRowAsGiven) {
    // Like torch, from_pretrained does not zero the padding row.
    const Embedding<float> emb = Embedding<float>::FromPretrained(NumberedTable(4, 3), 1);

    EXPECT_EQ(emb.PaddingIdx(), 1u);
    EXPECT_FLOAT_EQ((*emb.weight)(1, 0), 10.0f);
}

TEST(EmbeddingFromPretrained, AcceptsATransposedView) {
    const Tensor<float> transposed = NumberedTable(3, 4).Transpose();   // [4, 3] view
    ASSERT_FALSE(transposed.IsContiguous());

    const Embedding<float> emb = Embedding<float>::FromPretrained(transposed);
    const Tensor<float> out = emb.Forward(Ids({2}));

    // row 2 of the transpose is column 2 of the original: 2, 12, 22
    EXPECT_FLOAT_EQ(out(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(out(0, 1), 12.0f);
    EXPECT_FLOAT_EQ(out(0, 2), 22.0f);
}

TEST(EmbeddingFromPretrained, RejectsATableThatIsNot2D) {
    EXPECT_THROW(Embedding<float>::FromPretrained(Tensor<float>({2, 3, 4})), std::invalid_argument);
    EXPECT_THROW(Embedding<float>::FromPretrained(Tensor<float>({6})), std::invalid_argument);
}

// --- forward ----------------------------------------------------------------

TEST(EmbeddingForward, CopiesTheRowForEachIndex) {
    const Embedding<float> emb = Embedding<float>::FromPretrained(NumberedTable(5, 3));

    const Tensor<float> out = emb.Forward(Ids({0, 1, 2}));

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{3, 3}));
    for (size_t i = 0; i < 3; ++i)
        for (size_t d = 0; d < 3; ++d) EXPECT_FLOAT_EQ(out(i, d), static_cast<float>(i * 10 + d));
}

TEST(EmbeddingForward, RepeatedIndicesGiveTheSameRow) {
    const Embedding<float> emb = Embedding<float>::FromPretrained(NumberedTable(5, 3));

    const Tensor<float> out = emb.Forward(Ids({4, 1, 4}));

    for (size_t d = 0; d < 3; ++d) {
        EXPECT_FLOAT_EQ(out(0, d), 40.0f + d);
        EXPECT_FLOAT_EQ(out(2, d), 40.0f + d);
    }
}

TEST(EmbeddingForward, AppendsTheEmbeddingDimToAnyInputShape) {
    const Embedding<float> emb = Embedding<float>::FromPretrained(NumberedTable(10, 4));

    EXPECT_EQ(emb.Forward(Tensor<int64_t>({7}, 1)).Shape(), (std::vector<size_t>{7, 4}));
    EXPECT_EQ(emb.Forward(Tensor<int64_t>({2, 5}, 1)).Shape(), (std::vector<size_t>{2, 5, 4}));
    EXPECT_EQ(emb.Forward(Tensor<int64_t>({2, 3, 2}, 1)).Shape(), (std::vector<size_t>{2, 3, 2, 4}));
}

TEST(EmbeddingForward, BatchAndSequenceLayout) {
    const Embedding<float> emb = Embedding<float>::FromPretrained(NumberedTable(10, 2));
    Tensor<int64_t> ids({2, 3});
    ids(0, 0) = 1; ids(0, 1) = 2; ids(0, 2) = 3;
    ids(1, 0) = 7; ids(1, 1) = 8; ids(1, 2) = 9;

    const Tensor<float> out = emb.Forward(ids);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{2, 3, 2}));
    for (size_t b = 0; b < 2; ++b)
        for (size_t s = 0; s < 3; ++s)
            for (size_t d = 0; d < 2; ++d)
                EXPECT_FLOAT_EQ(out(b, s, d), static_cast<float>(ids(b, s) * 10 + static_cast<int64_t>(d)));
}

TEST(EmbeddingForward, AcceptsANonContiguousIndexTensor) {
    const Embedding<float> emb = Embedding<float>::FromPretrained(NumberedTable(10, 2));
    Tensor<int64_t> ids({2, 3});
    ids(0, 0) = 1; ids(0, 1) = 2; ids(0, 2) = 3;
    ids(1, 0) = 4; ids(1, 1) = 5; ids(1, 2) = 6;
    const Tensor<int64_t> view = ids.Transpose();   // [3, 2]
    ASSERT_FALSE(view.IsContiguous());

    const Tensor<float> out = emb.Forward(view);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{3, 2, 2}));
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_FLOAT_EQ(out(i, j, 0), static_cast<float>(view(i, j) * 10));
}

TEST(EmbeddingForward, WorksWithOtherIntegerIndexTypes) {
    const Embedding<float> emb = Embedding<float>::FromPretrained(NumberedTable(5, 2));

    EXPECT_FLOAT_EQ(emb.Forward(Tensor<int>({1}, 3))(0, 1), 31.0f);
    EXPECT_FLOAT_EQ(emb.Forward(Tensor<size_t>({1}, 4))(0, 0), 40.0f);
    EXPECT_FLOAT_EQ(emb.Forward(Tensor<int32_t>({1}, 2))(0, 1), 21.0f);
}

TEST(EmbeddingForward, ThePaddingIndexLooksUpNormally) {
    const Embedding<float> emb = Embedding<float>::FromPretrained(NumberedTable(5, 2), 0);
    EXPECT_FLOAT_EQ(emb.Forward(Ids({0}))(0, 1), 1.0f);
}

TEST(EmbeddingForward, RejectsAnIndexPastTheTable) {
    const Embedding<float> emb(5, 3);
    EXPECT_THROW(emb.Forward(Ids({0, 5})), std::out_of_range);
}

TEST(EmbeddingForward, RejectsANegativeIndex) {
    const Embedding<float> emb(5, 3);
    EXPECT_THROW(emb.Forward(Ids({-1})), std::out_of_range);
}

TEST(EmbeddingForward, RejectsAnEmptyIndexTensor) {
    const Embedding<float> emb(5, 3);
    EXPECT_THROW(emb.Forward(Tensor<int64_t>()), std::invalid_argument);
}

TEST(EmbeddingForward, TheOutputDoesNotAliasTheTable) {
    Embedding<float> emb = Embedding<float>::FromPretrained(NumberedTable(5, 3));
    Tensor<float> out = emb.Forward(Ids({2}));

    out(0, 0) = -1.0f;

    EXPECT_FLOAT_EQ((*emb.weight)(2, 0), 20.0f);
    EXPECT_FALSE(out.SharesStorage(*emb.weight));
}

TEST(EmbeddingForward, SeesChangesToTheWeight) {
    // A weight loader fills the table after construction.
    Embedding<float> emb(3, 2);
    (*emb.weight)(1, 0) = 5.0f;
    (*emb.weight)(1, 1) = 6.0f;

    const Tensor<float> out = emb.Forward(Ids({1}));

    EXPECT_FLOAT_EQ(out(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(out(0, 1), 6.0f);
}

TEST(EmbeddingForward, WorksWithDouble) {
    Embedding<double> emb(3, 2);
    emb.weight->Fill(0.25);

    const Tensor<double> out = emb.Forward(Ids({2, 0}));

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{2, 2}));
    EXPECT_DOUBLE_EQ(out(1, 1), 0.25);
}

// --- as a position embedding ------------------------------------------------

TEST(EmbeddingPositions, IndexingByPositionGivesTheFirstRowsInOrder) {
    const Embedding<float> positions = Embedding<float>::FromPretrained(NumberedTable(8, 2));

    const Tensor<float> out = positions.Forward(Ids({0, 1, 2}));

    for (size_t p = 0; p < 3; ++p) EXPECT_FLOAT_EQ(out(p, 0), static_cast<float>(p * 10));
}

TEST(EmbeddingPositions, TokenPlusPositionBroadcastsOverTheBatch) {
    const Embedding<float> tokens = Embedding<float>::FromPretrained(NumberedTable(10, 2));
    const Embedding<float> positions = Embedding<float>::FromPretrained(NumberedTable(4, 2));
    Tensor<int64_t> ids({2, 3}, 5);

    const Tensor<float> x = tokens.Forward(ids) + positions.Forward(Ids({0, 1, 2}));

    ASSERT_EQ(x.Shape(), (std::vector<size_t>{2, 3, 2}));
    EXPECT_FLOAT_EQ(x(1, 2, 1), 51.0f + 21.0f);   // token 5, position 2, dim 1
}

// --- Module interface -------------------------------------------------------

TEST(EmbeddingModule, ReportsParametersThroughABasePointer) {
    std::shared_ptr<Module<float>> m = std::make_shared<Embedding<float>>(5, 3);
    EXPECT_EQ(m->Parameters().size(), 1u);
}

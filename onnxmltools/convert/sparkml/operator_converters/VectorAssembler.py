from ...common._registration import register_converter


def convert_sparkml_vector_assembler(scope, operator, container):
    container.add_node('Concat', [s for s in operator.input_full_names],
                       operator.outputs[0].full_name, name=scope.get_unique_operator_name('Concat'), op_version=4, axis=1)


register_converter('pyspark.ml.feature.VectorAssembler', convert_sparkml_vector_assembler)
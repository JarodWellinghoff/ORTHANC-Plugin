import { Fragment } from "react";
import { metricTextParts } from "../../utils/resultComparison";

export default function FormattedMetricText({ value }) {
  return metricTextParts(value).map(({ key, tag: Tag, text }) => Tag
    ? <Tag key={key}>{text}</Tag>
    : <Fragment key={key}>{text}</Fragment>);
}

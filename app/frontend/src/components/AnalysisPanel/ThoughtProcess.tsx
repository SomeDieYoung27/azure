import { Stack } from "@fluentui/react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import json from "react-syntax-highlighter/dist/esm/languages/hljs/json";
import { a11yLight } from "react-syntax-highlighter/dist/esm/styles/hljs";

import styles from "./AnalysisPanel.module.css";

import { Thoughts } from "../../api";
import { TokenUsageGraph } from "./TokenUsageGraph";
import { AgentPlan } from "./AgentPlan";

SyntaxHighlighter.registerLanguage("json", json);

interface Props {
    thoughts: Thoughts[];
}

export const ThoughtProcess = ({ thoughts }: Props) => {
    return (
        <ul className={styles.tList}>
            {thoughts.map((t, ind) => {
                return (
                    <li className={styles.tListItem} key={ind}>
                        <div className={styles.tStep}>{t.title}</div>
                        <Stack horizontal tokens={{ childrenGap: 5 }}>
                            {t.props &&
                                (Object.keys(t.props).filter(k => k !== "token_usage" && k !== "query_plan") || []).map((k: any) => (
                                    <span className={styles.tProp} key={k}>
                                        {k}: {JSON.stringify(t.props?.[k])}
                                    </span>
                                ))}
                        </Stack>
                        {t.props?.token_usage && <TokenUsageGraph tokenUsage={t.props.token_usage} reasoningEffort={t.props.reasoning_effort} />}
                        {t.props?.query_plan && <AgentPlan query_plan={t.props.query_plan} description={t.description} />}
                        {Array.isArray(t.description) ? (
                            <SyntaxHighlighter language="json" wrapLongLines className={styles.tCodeBlock} style={a11yLight}>
                                {JSON.stringify(t.description, null, 2)}
                            </SyntaxHighlighter>
                        ) : (
                            <div>{t.description}</div>
                        )}
                    </li>
                );
            })}
        </ul>
    );
};